"""Functions to prepare an audio or video file or raw transcript.

For processing, an audio file is expected of a WAV format with a single channel
and a rate of 16kHz, which is what the function in this module creates.
"""

import os
import typing

import ffmpeg

EXTENSION = 'wav'
CHANNELS = 1
RATE = 16000


def format_audio(
    input_filepath: str,
    output_filepath: typing.Optional[str] = None,
    overwrite: typing.Optional[bool] = True,
) -> str:
    """Convert an audio or video file to a WAV file with 1 channel and 16 kHz.

    After conversion, the new audio file is stored on disk for reuse next time.
    The filename of the created WAV file is the input filepath with
    ``_processed.wav`` appended to it.

    Examples:
        >>> format_audio('example.mp4')
        'example.mp4_processed.wav'
        >>> format_audio('example2.mp3')
        'example2.mp3_processed.wav'

    Args:
        input_filepath: The audio or video file to convert.
        output_filepath: A filepath ending with ``.wav`` to store the result in.
            If this is not given, the ``input_filepath`` is with ``_processed.wav``
            appended to it and used.
        overwrite: Whether to overwrite the output file or not. Has default
            value ``True``.

    Returns:
        The filepath to which the created WAV file was written.

    Raises:
        ValueError: If the output filepath does not end in ``.wav``.
        ffmpeg.Error: in case of an error by ffmpeg.
        FileNotFoundError: If the input file was not found, or the output file
            was not created.
    """
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError('File {} not found.'.format(input_filepath))
    if output_filepath is None:
        output_filepath = input_filepath + '_processed.' + EXTENSION
    if not output_filepath.endswith(EXTENSION):
        raise ValueError('File {} does not have extension {}.'
                         .format(output_filepath, EXTENSION))
    try:
        ffmpeg \
            .input(input_filepath) \
            .output(output_filepath, ac=CHANNELS, ar=RATE) \
            .global_args('-hide_banner', '-loglevel', 'error') \
            .run(overwrite_output=overwrite)
    except FileNotFoundError as e:
        if 'No such file or directory: \'ffmpeg\'' in str(e):
            raise Exception('Could not find ffmpeg. Maybe ffmpeg is not installed?')
        raise
    if not os.path.isfile(output_filepath):
        raise FileNotFoundError('Could not find file {} after it should have been created.'
                                .format(output_filepath))
    return output_filepath

