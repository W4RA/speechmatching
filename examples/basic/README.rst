Basic example
#############

This basic example downloads a zip `file`_ with directories of recordings,
unpacks this in the current directory, and peforms various operations with the
sound recordings.

In "basic example 1", basic loading, transcript calculation, and similarity score
calculation is done. In "basic example 2", a small number of three groups of
recordings are created, with each having between 2 and 4 recordings. An final
recording is then separately loaded and matched with the three groups to see
which group matches the unknown recording best.

Finally, in "basic example 3", all downloaded recordings are loaded from
the directories. One group is created per directory, and all recordings from
that directory are added to that group.

A group is then taken to be tested with, and 20 random chosen recordings from
this group are matched with all groups, after which the best matching group is
chosen as matching group, or no decision can be made.

Please see the code in ``example.py`` for comments on how this works.

.. note::

    When processing the audio files in the downloaded ``recordings.zip`` file,
    they are also processed by the ``acoustic`` binary to have information
    extracted from them. This may take some time.

.. _file: https://zenodo.org/api/records/13284005/files/recordings.zip

Setup
*****

The basic example of how to use the speechmatching package can be run through
either Docker, or locally (manually).

Manual
======

It is advised to create a virtual environment using in a directory ``env/``::

    python3 -m venv env

and to load this with::

    source env/bin/activate

In the virtual environments (which you have activated two commands ago),
install the ``speechmatching`` package. From the two directories up with::

    (cd ../..; pip install .)

and make sure that the ``acoustic`` binary is already built using instructions
in the main ``README`` of this repository.

All requirements from ``requirements.txt`` need to be installed::

    pip install -r requirements.txt

And one should then be all set to run the script with::

    python3 main.py

Docker
======

Alternatively, Docker can be used, for which a Dockerfile is available.

Build the image using::

    docker build . -t speechmatching-basic

After this completes successfully, run the image using::

    docker run --rm -it speechmatching-basic

