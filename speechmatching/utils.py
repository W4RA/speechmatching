"""Various miscellaneous helper functions for the main code."""

import contextlib
import tempfile
import typing


@contextlib.contextmanager
def use_directory(
    directory: typing.Optional[str] = None
) -> typing.Iterator[str]:
    """Create a temporary directory or use an existing one.

    Args:
        directory: The directory to use or create. If this directory already
            exists, it is yielded, else a temporary directory is created.

    Yields:
        The path to the directory being used (either the one provided
        or the newly created temporary directory).
    """
    if directory is not None:
        yield directory
    else:
        temp = None
        try:
            temp = tempfile.TemporaryDirectory()
            yield temp.name
        finally:
            if temp is not None:
                temp.cleanup()


def ensure_list(
    data: typing.Any
) -> typing.List[typing.Any]:
    """Ensure the given data is of type :type:`list`, or make a new list.

    If the data is not yet a list, it is made into a list, and else it is
    returned as-is. Data of type :type:`tuple` will be converted into a
    :type:`list`, and else if the given ``data`` is not already a list, it will
    be returned as a list with ``data`` as single item in it.

    Examples:
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list((1, 2, 3))
        [1, 2, 3]
        >>> ensure_list("abc")
        ['abc']

    Args:
        data: The data to ensure is a list.

    Returns:
        The created or already given list.
    """
    if type(data) is tuple:
        return list(data)
    if type(data) is not list:
        return [data]
    return data


def dicts_to_tuples(data: dict) -> tuple:
    """Convert a dictionary to :type:`tuple`\ s.

    Conversion works iteratively, each key-value pair in the dictionary becomes
    a ``(key, value)`` :type:`tuple`.

    Examples:
        >>> d = {
        ...     'this': 'dictionary',
        ...     'is': {'nested': {'very': 'deep', 'extra': 'example'}},
        ...     'can': 'be',
        ...     'shallow': 'too'
        ... }
        >>> dicts_to_tuples(d)
        (('this', 'dictionary'),
         ('is', (('nested', (('very', 'deep'), ('extra', 'example'))),)),
         ('can', 'be'),
         ('shallow', 'too'))

    Args:
        data: The dictionary to convert.

    Returns:
        A tuple with nested tuples from the dictionary.
    """
    return tuple([
        (k, (dicts_to_tuples(v) if type(v) is dict else v))
        for k, v in data.items()
    ])


def process_string_alnum(s: str) -> str:
    """Normalize a string to only alphanumerical characters and no duplicates.

    Examples:
        >>> process_string_alnum("aaabbc!c???")
        'abc'

    Args:
        s: The string to process.

    Returns:
        The normalized string.
    """
    result = ''
    for c in s:
        if not c.isalnum():
            continue
        if len(result) > 0 and result[-1] == c:
            continue
        result += c
    return result


def print_docker_pull(messages: typing.Iterator[dict]):
    """Print the messages from a pull operation by ``docker-py``.

    The pull should be performed with ``stream=True`` and ``decode=True``, for
    the messages to be yielded in the right order.
    """
    for message in messages:
        if 'error' in message:
            raise Exception(message)
        if 'id' not in message:
            continue
        print(
            '{}: {} {}'.format(message['id'], message['status'], message.get('progress', '')),
            end=('\r' if 'progress' in message else '\n'),
            flush=True
        )

