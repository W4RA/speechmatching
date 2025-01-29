Speech Matching
###############

The ``speechmatching`` package allows for calculating similarity scores between short audio fragments containing speech. It then uses these similarity scores to find matches over multiple groups of recordings.

This is especially useful for rare languages from which not enough data exists to train a fully capable ASR model, or when the resources for training such a model are not available.

This package, along with the ``acoustic`` binary was created in combination with the thesis on Speech Matching [thesis]_ by Auke Schuringa for the University of Amsterdam and the Vrije Universiteit in the Netherlands.

.. warning::

    This project does not currently run on machines with the ARM64 architecture due to the ``intel-mkl`` library not supporting this.

.. [thesis] Auke Schuringa, "Energy-efficient phonetic matching for spoken words of rare indigenous languages in low-resource environments," Master's thesis, Vrĳe Universiteit Amsterdam and University of Amsterdam, Amsterdam, The Netherlands, Jan. 2025.

Setup
*****

The repository is made up of two parts. The first is the ``speechmatching`` package itself, and the second is the ``acoustic`` binary located in the ``acoustic/`` directory.

For details on setting up the ``acoustic`` package, please see its documentation. Because the ``acoustic`` package has specific dependencies and can be somewhat complex to build, it is recommended to run it using Docker.

The ``speechmatching`` Python package will by default look for the ``aukesch/speechmatching`` Docker image locally, and attempt to pull it if not available. For this, Docker needs to be installed.

Get Docker by installing it::

    curl https://get.docker.com/ | sudo sh

Then, grant permissions to the current user so Docker can be run without ``sudo``::

    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker

The ``speechmatching`` package requires ``ffmpeg`` for converting files with an audio stream to a usable format. Using ``apt``, this can be installed with::

    sudo apt-get update
    sudo apt-get install ffmpeg

Finally, install the ``speechmatching`` package from PyPI [package]_::

    pip install speechmatching

Or from the cloned repository::

    pip install .

Optionally, create a virtual environment first:

.. code-block:: bash

    sudo apt-get install python3-venv
    python3 -m venv env
    source env/bin/activate
    pip install .

.. [package] ``speechmatching`` on PyPI https://pypi.org/project/speechmatching/


Development
===========

During development, it may be useful to have a separate Docker image that is not named ``aukesch/speechmatching``, which can then contain changes made during development. To support this, the ``speechmatching`` package will look for a Docker image named ``speechmatching`` before it looks for the Docker image ``aukesch/speechmatching``. If the first docker image is available, the image under ``aukesch/`` will not be used.

Usage
*****

There are two ways to use the ``speechmatching`` package - either locally or from within Docker.

Local
=====

Running locally requires (automatically pulled) the Docker image containing the ``acoustic`` binary. This image sets the environment variable ``ACOUSTIC_RUNNING_IN_DOCKER=1``, which is not available when running outside the container.

When running ``speechmatching`` and an audio file needs to be processed, the package detects that it is not in Docker, then attempts to start a container from the ``aukesch/speechmatching`` Docker image (and attempt to pull it if it is not available locally). It will communicate with this container to process the audio file. When the program stops, the package tries to stop and remove the created container.

In other words, all that is needed is for Docker to be installed and set up correctly, and then the package can be run normally. These steps were explained in the previous section.

.. warning::

    Upon stopping or closing the program, one may see a note about "cleaning up." If this process is aborted, the Docker container may continue running even after the Python process closes.

    If that happens, the container can be manually identified and stopped::

        docker ps

    Copy the ``CONTAINER ID`` of the container in question, then stop and remove it with::

        docker stop CONTAINER_ID
        docker rm CONTAINER_ID


Docker
======

Alternatively, the ``speechmatching`` package entirely inside Docker. In this scenario, the package detects that the ``ACOUSTIC_RUNNING_IN_DOCKER=1`` environment variable is present and attempts to interact with the ``acoustic`` binary locally within the same container.

To do this, include a ``Dockerfile`` in the directory of the code with a structure like:

.. code-block::

    FROM aukesch/speechmatching
    COPY . .
    # more code...
    CMD ["python3", "main.py"]

After which the Docker image can be built and run.


Example
=======

Most of the functions that are needed to run this on a basic level are in the ``speechmatching.recording`` submodule.

In the following example, we assume the user has several audio files:

.. code-block:: text

    ./audio/speech1.mp3
    ./audio/speech2.3gp
    ./audio/speech3.wav
    ./audio/house1.mp3
    ./audio/house2.mp4
    ./audio/tree1.3gp
    ./audio/tree2.mp4
    ./audio/tree3.mp3
    ./audio/tree4.wav
    ./audio/unknown.mp3

The last file in this list is named ``unknown.mp3``, and it is believed this audio file belongs to one of the three spoken words ``speech``, ``house``, or ``tree``, for which we have several samples available.

A single recording can be loaded and analysed:

.. code-block:: python

    >>> from speechmatching.recording import Recording

    >>> # load the recording
    >>> speech1 = Recording('./audio/speech1.mp3')

    >>> # get the transcript
    >>> transcript = speech1.transcript

    >>> # print the most likely text
    >>> print(transcript.text)

    >>> # print the top likely texts
    >>> print(transcript.probable_texts())

    >>> # and calculate similarity scores to other recordings
    >>> speech2 = Recording('./audio/speech2.3gp')
    >>> print(speech2.similarity(speech1))
    >>> # same as
    >>> print(speech2.transcript.similarity(speech1.transcript))

These samples are sorted into groups:

.. code-block:: python

    >>> from speechmatching.recording import Group, Recording

    >>> groups = [
    ...     Group(
    ...         identifier='speech',
    ...         recordings=[
    ...             Recording('./audio/speech1.mp3'),
    ...             Recording('./audio/speech2.3gp'),
    ...             Recording('./audio/speech3.wav')
    ...         ]
    ...     ),
    ...     Group(
    ...         identifier='house',
    ...         recordings=[
    ...             Recording('./audio/house1.mp3'),
    ...             Recording('./audio/house2.mp4')
    ...         ]
    ...     ),
    ...     Group(
    ...         identifier='tree',
    ...         recordings=[
    ...             Recording('./audio/tree1.3gp'),
    ...             Recording('./audio/tree2.mp4'),
    ...             Recording('./audio/tree3.mp3'),
    ...             Recording('./audio/tree4.wav')
    ...         ]
    ...     )
    ... ]
                          
    >>> # load the unknown recording and match it
    >>> unknown = Recording('./audio/unknown.mp3')
    >>> match = unknown.match(groups)
    >>> print(match.identifier)
    >>> # best matching group is printed here

There are more possibilities, with much greater control over the process by using various arguments available to the functions around normalization of strings, calculating similarity scores, and strategies for finding the best matching group.

Using the function ``.match(...)`` of the ``Recording`` instance uses the combination of normalization and matching algorithms that was found to work best in the thesis for which this software was written.

More examples can be found in the ``examples/`` directory.

