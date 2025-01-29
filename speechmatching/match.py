"""Module containing the functions for calculating matching scores.

The functions help with creating the algorithms for normalization of string and
matching of algorithms, and combinations of the two.
"""

import copy
import functools
import typing

import jellyfish

from speechmatching.utils import ensure_list

MAP = {
    'soundex': 'soundex',
    'nysiis': 'nysiis',
    'metaphone': 'metaphone',
    'hamming': 'hamming_distance',
    'levenshtein': 'levenshtein_distance',
    'damerau': 'damerau_levenshtein_distance',
    'jaro': 'jaro_similarity',
    'winkler': 'jaro_winkler_similarity'
}

NORM_ALGS_TYPE_NORMED = typing.Dict[str, float]
NORM_ALGS_TYPE = typing.Union[
    str,
    typing.List[str],
    typing.Tuple[str, ...],
    NORM_ALGS_TYPE_NORMED
]

MATCH_ALGS_TYPE_NORMED = typing.Dict[str, NORM_ALGS_TYPE_NORMED]
MATCH_ALGS_TYPE = typing.Union[
    NORM_ALGS_TYPE,
    typing.Dict[str, NORM_ALGS_TYPE]
]


def ensure_norm_algs_dict(
    norm_algs: NORM_ALGS_TYPE
) -> NORM_ALGS_TYPE_NORMED:
    """Format the representation of a normalization algorithm dictionary.

    The given input is a representation of a dictionary describing the
    normalization algorithms and their factors in a final result.

    Examples:
        >>> ensure_norm_algs_dict('metaphone')
        {'metaphone': 1.0}
        >>> ensure_norm_algs_dict(['soundex', 'metaphone'])
        {'soundex': 0.5, 'metaphone': 0.5}
        >>> ensure_norm_algs_dict({'soundex': 0, 'metaphone': 0.1, 'nysiis': 0.3})
        {'soundex': 0.0, 'metaphone': 0.25, 'nysiis': 0.75}

    Args:
        norm_algs: The representation to make into a dictionary of
            normalization algorithms.

    Returns:
        The dictionary of normalization algorithms and their factors with sum
        up to 1.

    Raises:
        TypeError: If ``norm_algs`` is not a string, list, tuple, or dict.
    """
    if type(norm_algs) is str:
        norm_algs = [norm_algs]
    elif type(norm_algs) is tuple:
        norm_algs = list(norm_algs)
    if type(norm_algs) is list:
        norm_algs = {s: 1. for s in norm_algs}
    elif type(norm_algs) is not dict:
        raise TypeError('Unexpected type for norm_algs.')
    for k, v in norm_algs.items():
        if type(v) in (dict, list, tuple, str):
            norm_algs[k] = ensure_norm_algs_dict(v) # type: ignore
        else:
            total = sum(norm_algs.values())
            norm_algs = {
                k: v / total
                for k, v in norm_algs.items()
                if v > 0
            }
            break
    return norm_algs # type: ignore


def ensure_algs_dict(
    norm_algs: NORM_ALGS_TYPE,
    match_algs: MATCH_ALGS_TYPE
) -> typing.Tuple[
    NORM_ALGS_TYPE_NORMED,
    typing.Union[NORM_ALGS_TYPE_NORMED, MATCH_ALGS_TYPE_NORMED]
]:
    """Create a dictionary of normalization and matching algorithms.

    Please see the documentation of the :meth:`ensure_norm_algs_dict` function for
    how a representation of normalization algorithms is processed.

    The matching of audio files is performed using the results from the
    normalization algorithms. To construct the dictionary that describes how
    these matches are done, the normalization algorithm needs to be available.


    Examples:
        >>> norm_algs = {'metaphone': 0.25, 'nysiis': 0.7499999999999999}
        >>> ensure_algs_dict(norm_algs, 'hamming')[1]
        {'metaphone': {'hamming': 1.0}, 'nysiis': {'hamming': 1.0}}
        >>> ensure_algs_dict(norm_algs, ['hamming', 'jaro'])[1]
        {'metaphone': {'hamming': 0.5, 'jaro': 0.5}, 'nysiis': {'hamming': 0.5, 'jaro': 0.5}}
        >>> ensure_algs_dict(norm_algs, {'hamming': 0.1, 'jaro': 0.5})[1]
        {'metaphone': {'hamming': 0.16666666666666669, 'jaro': 0.8333333333333334}, 'nysiis': {'hamming': 0.16666666666666669, 'jaro': 0.8333333333333334}}
        >>> ensure_algs_dict(norm_algs, {'metaphone': {'hamming': 0.1, 'jaro': 0.3}, 'nysiis': {'levenshtein': 0.1, 'jaro': 0.2}})[1]
        {'metaphone': {'hamming': 0.25, 'jaro': 0.7499999999999999}, 'nysiis': {'levenshtein': 0.3333333333333333, 'jaro': 0.6666666666666666}}

    Args:
        norm_algs: The representation of normalization algorithms.
        match_algs: The representation of matching algorithms.

    Returns:
        A tuple of:
            - the dictionary of normalization algorithms, and
            - the dictionary of matching algorithms, where the dictionary of
              the matching algorithms has used information from the dictionary
              of normalization algorithms.
    """
    norm_algs = ensure_norm_algs_dict(norm_algs)
    match_algs = ensure_norm_algs_dict(match_algs)
    if not any(norm_alg in match_algs for norm_alg in norm_algs.keys()):
        match_algs = {norm_alg: copy.deepcopy(match_algs) for norm_alg in norm_algs}
    changed = False
    for norm_alg in list(norm_algs.keys()):
        if norm_alg not in match_algs:
            del norm_algs[norm_alg]
            changed = True
    if changed:
        norm_algs = ensure_norm_algs_dict(norm_algs)
    return norm_algs, match_algs


@functools.cache
def find_name(
    name: str,
    return_f: bool = True
) -> typing.Union[
    str,
    typing.Callable[[str, str], float],
    typing.Callable[[str], str]
]:
    """Find the function in the the :mod:`jellyfish` [jellyfish]_ module based on the name.

    This function is cached.

    The name for the functions and their function name in :mod:`jellyfish` are
    mapped as follows::

        {
            "soundex": "soundex",
            "nysiis": "nysiis",
            "metaphone": "metaphone",
            "hamming": "hamming_distance",
            "levenshtein": "levenshtein_distance",
            "damerau": "damerau_levenshtein_distance",
            "jaro": "jaro_similarity",
            "winkler": "jaro_winkler_similarity"
        }

    Args:
        name: The name to map against and find the :mod:`jellyfish` function for.
        return_f: Whether to return the found callable, or the string of the
            name of the function in :mod:`jellyfish`.

    Returns:
        The :mod:`jellyfish` function if ``return_f`` is ``True``, or the name
        of the function.

    Raises:
        ValueError: If the given name of an algorithm is not found, if multiple
            algorithms are found for a certain name, or if the name of an
            algorithm could not be matched for a given name.

    .. [jellyfish] Repository of jellyfish on GitHub https://github.com/jamesturk/jellyfish
    """
    match = None
    if hasattr(jellyfish, name):
        match = name
    is_found = False
    for k, v in MAP.items():
        if k in name:
            if is_found:
                raise ValueError('Multiple algorithms match {}.'.format(name))
            match = v
            is_found = True
    if not is_found:
        raise ValueError('Could not match name {} with an algorithm.'.format(name))
    if match is not None:
        if return_f:
            return getattr(jellyfish, match)
        return match
    raise ValueError('Algorithm {} not found.'.format(name))


@functools.lru_cache(maxsize=100000)
def normalize(s: str, name: typing.Optional[str] = None) -> str:
    """Normalize a string with a certain algorithm.

    This function is cached.

    The names for the algorithms that are supported are:
     - ``soundex`` for the Soundex algorithm
     - ``nysiis`` for the NYSIIS algorithm
     - ``metaphone`` for the Metaphone algorithm

    Args:
        s: The string to normalize.
        name: The name of the function to use. If not given, the string ``s`` is
            simply returned.

    Returns:
        The normalized string ``s`` or the string itself when no name of the
        algorithm to use was given.
    """
    if name is None:
        return s
    return find_name(name)(s)


@functools.lru_cache(maxsize=100000)
def match(
    normed1: str,
    normed2: str,
    match_algs: typing.Tuple[typing.Tuple[str, float]]
) -> typing.List[float]:
    r"""Calculate a matching score between two strings.

    The strings are expected to be normalized, however this is not strictly
    necessary. This function will not further normalize the given string,
    however.

    Args:
        normed1: The first string to compare.
        normed2: The second string to compare to the first string ``normed1``.
        match_algs: The matching algorithms to use and the factor with which
            they should count in the final result. An example of a correct
            value is::

                (
                    ('hamming', 0.3),
                    ('winkler', 0.5)
                )

            which notes that both the Hamming distance and the Jaro-Winkler
            similarities should be used, and the returned list of values will
            have the respective scores be multiplied by factors of
            respectively 0.3 and 0.5.

            If a distance function is given, we convert the result into a
            similarity score using similarity_score=1-(distance_score/max_length)

    Returns:
        A list of values of the result of match between the two given strings
        and each method with their factor.
    """
    max_length = max(len(normed1), len(normed2))
    normed1 = normed1.ljust(max_length, '\0')
    normed2 = normed2.ljust(max_length, '\0')
    result = []
    for match_name, match_factor in match_algs:
        name = find_name(match_name, return_f=False)
        assert type(name) is str
        val = getattr(jellyfish, name)(normed1, normed2)
        if name.endswith('distance'):
            if max_length == 0:
                val = 1
            else:
                val = 1 - val / max_length
        result.append(match_factor*val)
    return result

