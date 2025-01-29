"""Module containing global configuration for Docker and local based models."""

import os


class Config:
    """The configuration around interaction with the ``acoustic`` binary.

    Attributes:
        MODEL_DOCKER_IMAGE: The name(s) of the docker images with the
            ``acoustic`` binary to use. By default these are either:

             - ``speechmatching``: A locally build docker image to be used.

             - ``aukesch/speechmatching``: The docker image on docker.com,
                which can also be pulled.

        MODEL_DOCKER_BIN_LOCATION: The file path within the Docker
            container where the ``acoustic`` binary can be found.
        CACHE_DIR_LOCAL: The local directory path on the local machine
            used to store the models for the ``acoustic`` binary if not
            running with Docker.
        CACHE_DIR_DOCKER: The directory in the container used to store
            the models when using Docker for transcribing.
    """
    MODEL_DOCKER_IMAGE = [
        'speechmatching',
        'aukesch/speechmatching'
    ]
    MODEL_DOCKER_BIN_LOCATION = '/opt/bin/acoustic'
    CACHE_DIR_LOCAL = os.path.expanduser('~/.cache/acoustic')
    CACHE_DIR_DOCKER = '/root/.cache/acoustic'

config = Config()

