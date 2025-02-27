"""Functions and classes for interacting with the ``acoustic`` binary.

The ``acoustic`` binary is responsible for transcribing audio data into text
by using a neural network. Objects here allow for downloading these models,
caching them locally or for Docker usage, and running the binary either locally
or inside a Docker container.
"""

from __future__ import annotations

import abc
import atexit
import functools
import hashlib
import io
import os
import socket
import subprocess
import tarfile
import threading
import time
import typing

import docker
import requests

from speechmatching.audio import format_audio
from speechmatching.config import config
from speechmatching.utils import print_docker_pull


MODEL_URLS = {
    '70M': {
        'url': 'https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin',
        'hash': 'sha256:83ad885a65fadc3533fb81d21441ea770b2f8cd4e40d59ad356cc603e886775c'
    },
    '300M': {
        'url': 'https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin',
        'hash': 'sha256:2c59b6cd82a4dd0f90f4a635e58eca45584eb76275f3871a94ce90f04fa39490'
    }
}
TOKENS_URL = {
    'url': 'https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt',
    'hash': 'sha256:cc2b847964dd3001b76787253215b99b9ad1362820002faa408ace962eaed8e3'
}
CHUNK_SIZE = 1024 ** 2
DEFAULT_OUTPUT_FILENAME = 'temp_output.txt'
LOGS_FILEPATH = 'container_logs.txt'


def get_cache_filepath(
    location: str,
    make_dir: bool = True,
    docker: bool = False
) -> str:
    """Get the filepath for a URL or file in the cache dir.

    The default cache dir can be found in the :class:`speechmatching.config.Config` class.

    Args:
        location: The URL or file for which to construct the cache filepath.
        make_dir: Make the target directory if it does not exist. By default
            valued ``True``.
        docker: Whether to use the cache dir in the Docker environment, or in
            the local environment. By default set to ``False``, which means the
            local environment is used.

    Returns:
        A string representing the full path to the cached file. For URLs, only
        the filename portion (after the last "/") is used in the cache path.
    """
    if location.startswith('http://') or location.startswith('https://'):
        location = location.rsplit('/', 1)[1]
    if docker:
        cache_dir = config.CACHE_DIR_DOCKER
    else:
        cache_dir = config.CACHE_DIR_LOCAL
    if make_dir and not docker and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, location)


def download_files(
    model_size: str = '70M',
    overwrite: bool = False
):
    """Download the tokens URL [tokens]_ and a certain model URL to the cache dir.

    The model is either one of 70 million [70model]_ or 300 million [300model]_
    parameters.

    These files are required as input to the ``acoustic`` binary to transcribe
    an audio file to characters and their probabilities.

    Args:
        model_size: The model to download. This can be either ``70M`` or ``300M``.
            The default value is ``70M``.
        overwrite: Whether to overwrite an existing file or not.

    Raises:
        AssertionError: If the HTTP response code is not 200 or if the hash
            digest of the downloaded file does not match the expected digest.
    """
    #.. [tokens] https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt
    #.. [70model] https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin
    #.. [300model] https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin

    for url_data in (
        TOKENS_URL,
        MODEL_URLS[model_size]
    ):
        url = url_data['url']
        hash_algorithm, hash_digest = url_data['hash'].split(':', 1)
        filepath = get_cache_filepath(url)
        filepath_bak = filepath + '.bak'
        if os.path.isfile(filepath):
            if not overwrite:
                continue
            os.rename(filepath, filepath_bak)
        with open(filepath, 'wb') as f:
            h = hashlib.new(hash_algorithm)
            response = requests.get(url, stream=True)
            assert response.status_code == 200
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk is not None:
                    h.update(chunk)
                    f.write(chunk)
            assert h.hexdigest() == hash_digest
            #assert f.tell() == int(response.headers['Content-Length'])
        if os.path.isfile(filepath_bak):
            os.remove(filepath_bak)


class Model(abc.ABC):
    """Abstract base class representing a model for transcribing audio.

    Args:
        model_size: The model size to use. Either ``70M`` or ``300M``. See
            documentation for function :meth:`download_files` for more
            information.
    """
    PROCESSES: typing.List[Model] = []

    def __init__(self, model_size: str = '70M'):
        self._model_size = model_size
        self.start()
        self.record_process(self)

    @classmethod
    def record_process(cls, instance: Model):
        """Add a running model to keep track of.

        It is important to keep track of the instances of the model that are in
        use in order to stop these instances when the program closes.

        Args:
            instance: The model to register.
        """
        cls.PROCESSES.append(instance)

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self, message: str):
        pass

    @abc.abstractmethod
    def read_result(self, filepath: str) -> str:
        pass


class LocalModel(Model):
    """A model that runs the ``acoustic`` binary locally on the host machine.

    This model can be used when the ``acoustic`` binary and the required
    libraries have been compiled and are available locally. This is the case
    when this code is run in Docker with the compiled binary, or when effort
    has been put in to compile the binary locally outside of Docker.

    When not running in Docker from the ``aukesch/speechmatching`` image, this
    model should likely not be used, and the :class:`DockerModel` should be
    used instead.

    Upon initialization of an instance, the required files for transcribing are
    downloaded if not downloaded yet.

    Args:
        acoustic_location: Optional. This is the location of the ``acoustic``
            binary if this is different from one of the expected locations.
        *args, **kwargs: See the initialization arguments of :class:`Model`\ .
    """

    def __init__(self, *args, acoustic_location: typing.Optional[str] = None,
                 **kwargs):
        self._acoustic_location = acoustic_location
        super().__init__(*args, **kwargs)
        download_files(model_size=self._model_size)

    @functools.cached_property
    def _process(self):
        """Create the process for transcribing.

        If the location of the ``acoustic`` model was not given upon
        initialization of the instance, an attempt is made to locate the
        binary in one of the following locations:

         - ``/opt/bin/acoustic``
         - ``/usr/local/bin/acoustic``
         - ``acoustic`` (in the local directory)

        If the binary is found, the process is started and waits for input over
        standard input for files to transcribe.

        Returns:
            A :class:`subprocess.Popen` instance that communicates with the
            ``acoustic`` binary over standard input and output.

        Raises:
            Exception: In case the binary is not found.
        """
        if self._acoustic_location is not None:
            possible_locations = (
                self._acoustic_location,
            )
        else:
            possible_locations = (
                '/opt/bin/acoustic',
                '/usr/local/bin/acoustic',
                'acoustic'
            )
        for filepath in possible_locations:
            if not os.path.isfile(filepath):
                continue
            print(f"Using acoustic binary at: {filepath}")
            return subprocess.Popen(
                [' '.join([
                    filepath,
                    '-stdin',
                    '-o', os.path.join(config.CACHE_DIR_LOCAL, DEFAULT_OUTPUT_FILENAME),
                    '-t', get_cache_filepath(TOKENS_URL['url']),
                    '-am', get_cache_filepath(MODEL_URLS['70M']['url'])
                ])],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                shell=True,
                #preexec_fn=os.setsid
            )
        raise Exception('acoustic binary not found in any given or known location.')

    def start(self):
        """Start the model by creating the process to transcribe with."""
        _ = self._process

    def stop(self):
        """Stop the model by terminating the running process."""
        if '_process' in self.__dict__:
            self._process.terminate()
            self._process.wait()
            del self._process

    def read(self) -> str:
        """Read a line from the process over standard output.

        Returns:
            A single stripped line of output.
        """
        while True:
            line = self._process.stdout.readline()
            if len(line) == 0:
                continue
            return str(line, 'utf8').strip()

    def write(self, message: str):
        """Write a message to the process over standard input.

        Args:
            message: The message to write.
        """
        self._process.stdin.write(bytes(message+'\n', 'utf8'))
        self._process.stdin.flush()

    def read_result(self, filepath: typing.Optional[str] = None) -> str:
        """Read output from the transcribing process.

        Args:
            filepath: The filepath to read from. This is by default the default
                file the ``acoustic`` binary writes results to, which is
                defined in the :class:`speechmatching.config.Config` class.

        Returns:
            A line of output from standard output.
        """
        if filepath is None:
            filepath = os.path.join(
                config.CACHE_DIR_LOCAL,
                DEFAULT_OUTPUT_FILENAME
            )
        with open(filepath, 'r') as f:
            return f.read()


class DockerModel(Model):
    """A model that runs the ``acoustic`` binary in a Docker container.

    This model communicates with a container of the Docker image under
    (normally) name ``aukesch/speechmatching`` or ``speechmatching``. The model
    starts the container, interacts with it, and stops it when asked to.

    This is the model that should be used when the ``acoustic`` binary and its
    libraries have not been compiled locally, but are only available through
    Docker, and the software using the ``speechmatching`` package does not run
    in Docker.

    Args:
        pull_image: Whether to pull the ``aukesch/speechmatching`` Docker image
            when a local image or this one is not found.
        *args, **kwargs: See the initialization arguments of :class:`Model`\ .
    """

    def __init__(self, *args, pull_image: bool = True, **kwargs):
        self._stop_alive_checker = False
        self._pull_image = pull_image
        super().__init__(*args, **kwargs)

    @functools.cached_property
    def _client(self) -> docker.APIClient:
        """Create the Docker API client for managing containers.

        Returns:
            A :class:`docker.APIClient` instance.

        Raises:
            Exception: If the client could not be started, which is an
                indication that Docker is not installed. Or, if the Docker could
                not be used due to permission problems.
        """
        try:
            return docker.APIClient()
        except docker.errors.DockerException as e:
            inner_error_type = type(e.__cause__.args[0].args[1])
            if inner_error_type is FileNotFoundError:
                raise Exception('Docker could not be found.'
                                ' Maybe Docker is not installed?'
                                ' Please see the README.')
            if inner_error_type is PermissionError:
                raise Exception('No permission to use Docker.'
                                ' Maybe Docker is not setup to run without root?'
                                ' Please see the README.')

    @functools.cached_property
    def _image(self) -> str:
        """Find or pull the image to use for the ``acoustic`` binary.

        The names for the Docker images listed in
        :class:`speechmatching.config.Config` are checked in order, and if one
        is found to exist, it is returned.

        If no images are found to exist, the first image name containing ``/`` is
        pulled if ``pull_image`` was not set to ``False`` on initialization of
        this class. After pulling, the name of the pulled Docker image is
        returned.

        Returns:
            An existing Docker image.

        Raises:
            Exception: If no Docker image could be found and no Docker image
                should be pulled because ``pull_image`` was set to ``False``,
                or if no Docker image name could be found that can be pulled.
        """
        images = config.MODEL_DOCKER_IMAGE
        if type(images) is not list:
            images = [images]
        for name in images:
            try:
                print('Trying to find Docker image {}...'.format(name), end='', flush=True)
                _ = self._client.inspect_image(name)
                print(' found.', flush=True)
            except docker.errors.ImageNotFound:
                print(' not found.', flush=True)
                continue
            return name
        if not self._pull_image:
            raise Exception('Could not find a usable Docker image.')
        for to_pull in images:
            if '/' in to_pull:
                try:
                    print('Pulling Docker image {}...'.format(name), end='', flush=True)
                    print_docker_pull(self._client.pull(to_pull, tag='latest',
                                                        stream=True, decode=True))
                    print(' done.', flush=True)
                except docker.errors.ImageNotFound:
                    print(' failed.', flush=True)
                    continue
                break
        else:
            raise Exception('Could not find a pullable Docker image.')
        return to_pull

    @functools.cached_property
    def _container(self) -> dict:
        """Starts the container containing the ``acoustic`` binary.

        The container is started with the command to start the ``acoustic``
        binary and let it wait for input over standard input for audio files
        to transcribe.

        Returns:
            Information about the container created by the client.
        """
        container = self._client.create_container(
            self._image,
            stdin_open=True,
            command='{} -stdin -t {} -o {} -am {}'.format(
                config.MODEL_DOCKER_BIN_LOCATION,
                get_cache_filepath(TOKENS_URL['url'], docker=True),
                os.path.join(config.CACHE_DIR_DOCKER, DEFAULT_OUTPUT_FILENAME),
                get_cache_filepath(MODEL_URLS[self._model_size]['url'], docker=True)
            )
        )
        self._client.start(container)
        self._alive_checker_thread = threading.Thread(
            target=self._alive_checker,
            args=(container,),
            daemon=True
        )
        self._alive_checker_thread.start()
        return container

    def _alive_checker(self, container: dict, interval: int = 2):
        """Continuously check if the container is still alive.

        If the container is not alive anymore, the logs of the container are
        printed to the ``LOGS_FILEPATH`` global variable filepath, and the
        program is exited with exit code 1.

        Args:
            container: The container to be checked.
            interval: The interval in seconds with which should be checked,
                default is 2.
        """
        while True:
            if self._stop_alive_checker:
                break
            if not self._client.inspect_container(container)['State']['Running']:
                with open(LOGS_FILEPATH, 'wb') as f:
                    f.write(self._client.logs(container))
                print('The Docker container stopped running. Logs dumped to {}.'
                      .format(LOGS_FILEPATH))
                os._exit(1)
            time.sleep(interval)

    @functools.cached_property
    def _container_stdout(self) -> socket.SocketIO:
        """The socket for standard output from the running container.

        Returns:
            The socket to the container to read information from.
        """
        return self._client.attach_socket(
            self._container,
            params={
                'stdout': 1,
                'stream': 1
            }
        )

    @functools.cached_property
    def _container_stdout_lines(self) -> typing.Iterator[str]:
        """Iterator over lines from the standard output socket.

        Yields:
            A line read from standard output from the container.
        """
        while True:
            line = next(self._container_stdout)
            for s in str(line, 'utf8', errors='ignore').split('\n'):
                if len(s) == 0:
                    continue
                yield s

    @functools.cached_property
    def _socket(self) -> socket.SocketIO:
        """The socket for standard input to the running container.

        Returns:
            The socket to the container to write information to.
        """
        return self._client.attach_socket(
            self._container,
            params={
                'stdin': 1,
                'stream': 1
            }
        )

    def _stop_client(self):
        """Stop the Docker communication client."""
        if '_client' in self.__dict__:
            del self._client

    def _stop_container(self):
        """Stop and remove the running container gracefully."""
        if '_container' in self.__dict__:
            self._stop_alive_checker = True
            self._alive_checker_thread.join()
            self._stop_socket()
            self._client.stop(self._container)
            self._client.wait(self._container)
            self._client.remove_container(self._container)
            del self._container
            if '_container_stdout' in self.__dict__:
                del self._container_stdout

    def _stop_socket(self):
        """Close the sockets to the container."""
        for socket_s in (
            '_socket',
            '_container_stdout'
        ):
            if socket_s in self.__dict__:
                self._socket.close()
                del self._socket

    def _copy_file(self, src_filepath: str, dst_dirpath: str) -> str:
        """Copy a file from the host to the container.

        After copying, the file becomes available in the container and to the
        ``acoustic`` binary for further processing. However, the binary still
        needs to be sent the message that it should process the copied file.

        If a file already exists at the location the local file will be copied
        to, this file will be overwritten.

        Args:
            src_filepath: Path to the file on the host machine.
            dst_dirpath: Path to copy the file from ``src_filepath`` to
                on the container.

        Returns:
            The location in the container the file has been copied to.
        """
        stream = io.BytesIO()
        name = 'audio.wav'
        with open(src_filepath, 'rb') as f, \
            tarfile.open(fileobj=stream, mode='w|') as tf:
            info = tf.gettarinfo(fileobj=f)
            info.name = name
            tf.addfile(info, f)
        self._client.put_archive(self._container, dst_dirpath, stream.getvalue())
        return os.path.join(dst_dirpath, name)

    def start(self):
        """Start the container and create a socket for writing to it."""
        _ = self._socket

    def stop(self):
        """Stop the sockets, container and Docker client."""
        self._stop_socket()
        self._stop_container()
        self._stop_client()

    def write(self, message: str):
        """Write a message to the container over standard input.

        Args:
            message: The message to write.
        """
        self._socket._sock.send(bytes(message+'\n', 'utf8')) # type: ignore

    def read(self) -> str:
        """Read a line from the container over standard output.

        Returns:
            A single line of output.
        """
        return next(self._container_stdout_lines)

    def read_result(self, filepath: typing.Optional[str] = None) -> str:
        """Read output from the transcribing container.

        The file written by the ``acoustic`` binary is copied into the host
        machine and then read, and is not being read directly from the
        container.

        Args:
            filepath: The filepath to read from. This is by default the default
                file the ``acoustic`` binary writes results to in the
                container, which is defined in the
                :class:`speechmatching.config.Config` class.

        Returns:
            The result from the model after processing an audio file.

        Raises:
            ValueError: If the file copied from the container is somehow not
                found.
        """
        if filepath is None:
            filepath = os.path.join(
                config.CACHE_DIR_DOCKER,
                DEFAULT_OUTPUT_FILENAME
            )
        stream, _ = self._client.get_archive(self._container, filepath)
        with tarfile.open(fileobj=io.BytesIO(b''.join(stream)), mode='r:') as tf:
            f = tf.extractfile(os.path.basename(filepath))
            if f is None:
                raise ValueError('Could not load {} from tar stream.'.format(filepath))
            return str(f.read(), 'utf8')


class Transcriptor:
    """The transcriptor for guiding the transcribing process.

    Args:
        model_location: The model to use, can have either value ``local``, or
            value ``docker``. If not set, the value will be attempted to be
            determined by looking for the ``ACOUSTIC_RUNNING_IN_DOCKER``
            environment variable set by the ``aukesch/speechmatching`` or
            ``speechmatching`` Docker images, if one of them is being used.
    """

    def __init__(self, model_location: typing.Optional[str] = None):
        if model_location is None:
            if 'ACOUSTIC_RUNNING_IN_DOCKER' in os.environ:
                model_location = 'local'
            else:
                model_location = 'docker'
        if model_location == 'local':
            self._model = LocalModel()
        elif model_location == 'docker':
            self._model = DockerModel()
        self._lock = threading.Lock()

    def _transcribe(self, input_filepath: str) -> str:
        """Run the process described in function :meth:`Transcriptor.transcribe` in this class."""
        input_filepath = format_audio(input_filepath)
        # maybe copy the file to the Docker container
        if isinstance(self._model, DockerModel):
            input_filepath = self._model._copy_file(
                input_filepath,
                config.CACHE_DIR_DOCKER
            )
        # wait until the container signals it is ready for processing a new
        # audio file
        while True:
            line = self._model.read()
            if line == "Waiting for WAV file path.":
                break
        # let the running ``acoustic`` binary know it needs to process this
        # filepath
        self._model.write(input_filepath)
        # wait for the model to notify the file has been processed and return
        # result.
        while True:
            line = self._model.read()
            if line.startswith('Writing file '):
                output_filepath = line.split(' ', 2)[2].rstrip('.')
            if line.startswith('Processed file '):
                return self._model.read_result(output_filepath)

    def transcribe(self, input_filepath: str) -> str:
        """Transcribe an audio or video file using the running model.

        This process can only run once at a time for each running instance of
        the model.

        The given audio or video file is first processed into a WAV file of a
        single channel and 16000 hertz. If the Docker container is running, the
        file is copied into the running container. The file is then processed
        using the running ``acoustic`` binary, and the result is retrieved when
        the signal is given that processing has finished. This signal is given
        by writing on standard output from the process in the Docker container.

        Args:
            input_filepath: The audio or video file to process.

        Returns:
            The raw output from the transcribing process in the form described
            in the documentation of class
            :class:`speechmatching.recording.Transcript`.
        """
        with self._lock:
            return self._transcribe(input_filepath)

    def stop(self):
        """Stop the model."""
        self._model.stop()


@atexit.register
def stop():
    """Stop all running models, both local and in Docker containers.

    This method is registered to be automatically called when the signal is
    given to stop. It will use the processes stored in the global class
    variable in the :class:`Model` class to determine which processes are running
    and need to be stopped.

    This function may take several seconds to complete.

    .. warning::

        If this process is stopped before it finishes, and a Docker container
        was used for transcribing, the container may not be stopped correctly
        and it needs to be stopped manually.
    """
    print('Cleaning up... do not cancel.')
    print('Stopping', len(Model.PROCESSES), 'process(es):')
    for i, process in enumerate(Model.PROCESSES):
        print('Stopping process', process, flush=True)
        process.stop()

