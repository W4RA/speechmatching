acoustic binary
###############

The ``acoustic`` binary processes audio files as input using an acoustic model for a neural network. It maps outcomes to probabilities for tokens.

Installation
************

To compile the ``acoustic`` binary, several dependencies with specific versions are required. Due to the somewhat complex setup, it is recommended to use Docker.

Docker
======

To compile the ``acoustic`` binary and the ``speechmatching`` package into a Docker image, use the ``Dockerfile`` provided in the repositories root directory. Build the Docker image under the name ``flashlight-acoustic`` so the image can be easily accessed by the ``speechmatching`` package:

.. code-block:: bash

    docker build . -t flashlight-acoustic

This process may take several minutes to complete.

Manual
======

The following versions of the dependencies were found to work well. It is possible other versions are also supported, but this was not thoroughly tested:
 - ArrayFire [arrayfire]_ version ``3.8.3`` with options
    - ``AF_BUILD_CPU=ON``
    - ``AF_BUILD_CUDA=OFF``
    - ``AF_BUILD_OPENCL=OFF``
    - ``AF_BUILD_EXAMPLES=OFF``
    - ``AF_WITH_IMAGEIO=OFF``
    - ``BUILD_TESTING=OFF``
    - ``AF_BUILD_DOCS=OFF``
 - oneDNN [onednn]_ version ``2.5.2`` with options
    - ``DNNL_BUILD_EXAMPLES=OFF``
 - Gloo [gloo]_ branch ``56b221c0a811491d2dc2a3254b468ad687bbdaab`` with options
    - ``USE_MPI=ON``
 - kenlm [kenlm]_ branch ``9af679c38477b564c26917a5dcf52d2c86177fb9`` with options
    - ``CMAKE_POSITION_INDEPENDENT_CODE=ON``
 - Flashlight [flashlight]_ version ``0.3.1`` with options
    - ``FL_BACKEND=CPU``
    - ``FL_BUILD_ALL_APPS=OFF``
    - ``FL_BUILD_PKG_TEXT=ON``
    - ``FL_BUILD_PKG_RUNTIME=ON``
    - ``FL_BUILD_PKG_SPEECH=ON``
    - ``FL_BUILD_TESTS=OFF``
    - ``FL_BUILD_EXAMPLES=OFF``
    - ``FL_BUILD_APP_ASR=ON``
    - ``FL_BUILD_APP_ASR_TOOLS=ON``
 - Library ``intel-mkl-64bit-2020.4-912`` from Intel repos [intelrepos]_ as:

    .. code-block:: bash

        wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        sh -c 'echo deb [trusted=yes] https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
        apt-get install -y --no-install-recommends intel-mkl-64bit-2020.4-912

All of the above should also be combined with option ``CMAKE_BUILD_TYPE=Release``.

The ``acoustic`` binary itself should can then be made and installed by:

.. code-block:: bash

    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_MODULE_PATH=/usr/local/share/flashlight/cmake
    make
    mkdir /opt/bin
    cp acoustic /opt/bin

Please see the ``Dockerfile`` in the main directory of the repository for more detailed information.

.. [arrayfire] Repository of ArrayFire on Github https://github.com/arrayfire/arrayfire
.. [onednn] Repository of oneDNN on GitHub https://github.com/oneapi-src/onednn
.. [gloo] Repository of Gloo on GitHub https://github.com/facebookincubator/gloo
.. [kenlm] Repository of kenlm on GitHub https://github.com/kpu/kenlm
.. [flashlight] Repository of Flashlight on GitHub https://github.com/flashlight/flashlight
.. [intelrepos] Intel repository https://apt.repos.intel.com/mkl

Usage
*****

The acoustic binary can be run with the following arguments:

--help  Show the help message.

-i PATH, --input-filepath PATH    Path to the WAV file to process, which should have a single channels and 16000 hertz.

--am PATH, --acoustic-model-filepath PATH        Path to the acoustic model to use, this is usually either the
 - 70 million parameter model [70model]_,
 - or the 300 million parameter model [300model]_.

-o PATH, --output-filepath PATH             Path to the file to write the probabilities to. If the options ``-stdin`` is used, this file is overwritten whenever a new WAV file is given for processing.

-t PATH, --tokens-filepath PATH             Path to the file with tokens to use. Usually this is a default tokens file [tokens]_.

-stdin, --standard-input                    Listen to standard input for filenames of WAV files to process. If this option is used together with the ``-i path``, ``--input-filepath PATH`` option, then the WAV file given over the command line is processed first, after which the program continues with and start listening on standard input.

.. [70model] Acoustic model with 70 million parameters https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin (archived at https://web.archive.org/web/20230603172217id\_/https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin)
.. [300model] Acoustic model with 300 million parameters https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin (archived at https://web.archive.org/web/20220729100332id\_/https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin)
.. [tokens] Default tokens for both acoustic models https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt (archived at https://web.archive.org/web/20220729100540id\_/https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt)

``speechmatching`` package
~~~~~~~~~~~~~~~~~~~~~~~~~~

When compiled into a Docker image under name ``flashlight-acoustic``, the ``speechmatching`` package can be installed locally with ``pip``, and it will look for and start a container for the ``flashlight-acoustic`` image. When the program is stopped, the package will attempt to stop and remove any created container.

See the documentation for the ``speechmatching`` package for further information.

