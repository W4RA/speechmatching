"""The functions and classes around storing, processing and handling audio.

These are the main objects that are likely used when using package
``speechmatching`` in code. Please see the README for instruction on how to use
this, and have a look at the examples.
"""

from __future__ import annotations

import functools
import json
import os
import random
import sys
import typing

import magic
import numpy as np

from speechmatching.match import (
    match,
    normalize,
    ensure_algs_dict,
    NORM_ALGS_TYPE,
    MATCH_ALGS_TYPE
)
from speechmatching.model import Transcriptor
from speechmatching.utils import (
    ensure_list,
    process_string_alnum,
    dicts_to_tuples
)

NORM_ALGS_BEST_PERFORMING = ['soundex', 'nysiis', 'metaphone']
MATCH_ALGS_BEST_PERFORMING = ['damerau', 'jaro', 'winkler']


def sanitize_raw_transcript(
    transcript: str,
    no_dup: bool = True,
    no_space: bool = True
) -> str:
    """Sanitize a raw transcript by removing certain characters.

    The character ``|`` is treated as a special character and will be transformed
    into a space if ``no_space`` is set to ``False``. Optionally, duplicate
    characters can be deleted, which is set to ``True`` by default.

    Examples:
        >>> sanitize_raw_transcript("he||llo||world!")
        'heloworld'
        >>> sanitize_raw_transcript("he||llo||world!", no_space=False)
        'he lo world'

    Args:
        transcript: The raw transcript.
        no_dup: Whether to remove duplicate characters or not. Default value
            is ``True``.
        no_space: Whether to remove spaces or not. Default value is ``True``.

    Returns:
        The final sanitized string.
    """
    result = ''
    for c in transcript:
        new_char = None
        if c == '|':
            if not no_space:
                new_char = ' '
        elif c.isalnum():
            new_char = c
        else:
            continue
        if new_char is None:
            continue
        if no_dup and len(result) > 0 and result[-1] == new_char:
            continue
        result += new_char
    return result.strip()


class Transcript:
    """The handling of raw output for a transcript.

    The raw transcript is in a form as shown the description of the
    ``raw`` variable below, and usually comes from the
    :class:`speechmatching.model.Transcriptor` used in the :class:`Recording`
    class.

    This class performs various operations on the raw data, like extracting
    probable texts, and calculating matching scores with different transcripts.

    Args:
        raw: The string with multiple lines for the probabilities of
            the characters in the transcript. The first line in the string
            should note the characters, and each subsequent line represents
            the probabilities of these characters for the specific time
            step. This is formatted as::

                char1 char2 char3 [...] charx
                prob1 prob2 prob3 [...] probx
                [...] (time steps)
                prob1 prob2 prob3 [...] probx

            for which an example is::

                a b c [...] z
                0.1 0.3 0.05 [...] 0.01
                [...]
                0.01 0.8 0.1 [...] 0.001
    """

    def __init__(
        self,
        raw: str
    ):
        self._cached: typing.Dict[
            typing.Tuple[float, bool, bool, bool],
            typing.Dict[str, float]
        ] = {}
        self._raw = raw

    @functools.cached_property
    def probabilities(self) -> np.typing.NDArray:
        """Create the list of lists of probabilities per time step.

        The first line of the raw input is left out of the returned data, and
        only the probabilities are included from the raw data.

        Returns:
            The matrix of probabilities with the time step on the y-axis
            and the character probability on the x-axis.
        """
        return np.array([
            [float(d) for d in line.strip().split(' ')]
            for line in self._raw.strip().splitlines()[1:]
        ])

    @property
    def tokens(self) -> typing.List[str]:
        """The list of characters in the raw data in order.

        Returns:
            The characters in the first line of the raw data in the same order
            they have in this first line.
        """
        return self._raw.split('\n', 1)[0].strip().split(' ')

    @functools.cached_property
    def text(self) -> str:
        """Calculate the text with most probable character in each time step.

        After calculation of the raw string, it is still processed with the
        :meth:`speechmatching.utils.process_string_alnum` function before
        being returned. The characters are uppercased before returning.

        Returns:
            The most probable text by selecting the most likely token at each
            time step.
        """
        tokens = np.array(self.tokens)
        return process_string_alnum(
            ''.join([c for c in tokens[np.argmax(self.probabilities, axis=1)]])
        ).upper()

    def probable_texts(
        self,
        min_probability: float = 0.2,
        cache: bool = True,
        normalize: bool = True,
        or_best: bool = False,
        or_best_char: bool = False,
    ) -> typing.Dict[str, float]:
        """Calculate probable texts from the raw data and their probabilities.

        The raw data given to the instance has probabilities at each time step
        for given characters. With these probabilities, probable texts or
        strings of characters can be calculated. This can be done by taking
        all characters in each time step with a probability above the minimum
        probability and combining these into multiple strings.

        It can happen that multiple characters in a time step have a sufficient
        probability, in which case multiple string are calculated. As an
        example, if there are three time steps, where time step one has 1
        character of sufficient probability, time step two has 3, and time step
        three has 2 such characters, then a total of :math:`1 \cdot 3\cdot 2=6`
        texts will be calculated.

        It can happen that none of the characters in a time step have a
        sufficient probability, in which case what happens can be controlled
        using the arguments ``or_best`` and ``or_best_char``.

        If ``or_best`` and ``or_best_char`` are both set to ``False``, it is possible
        for a single empty string to be returned in the dictionary.

        Args:
            min_probability: The minimum probability for a token to be
                considered. Has a default value of 0.2.
            cache: Use the cache to store the result for the given set of
                arguments, or to retrieve the result from there and return
                it.
            normalize: Normalize the probabilities for the returned texts to 1.
            or_best: If no text could be extracted due to all character not
                exceeding the minimum probability, simply take the most likely
                text, which is the text return by the `text` property of the
                instance.
            or_best_char: if no character for a time step could be selected
                due to none of them exceeding the minimum probability, use the
                character from the time step with the highest probability.

        Returns:
            The most likely texts according to the given arguments in the form
            of a dictionary::

                {
                    "TEXTA": 0.3,
                    "TEXTB": 0.2,
                    "TEXTC": 0.5
                }

            where the probabilities add up to 1 if ``normalize`` is set to
            ``True``.
        """
        key = (min_probability, normalize, or_best, or_best_char)
        if cache and key in self._cached:
            return self._cached[key]
        # the texts are in the form of a list of tuples, with each tuple
        # containing a string, and a probability.
        texts = [('', 1.)]
        for timestep in self.probabilities:
            # new texts for a time step are calculated using the previous
            # texts, and if any new texts have been found, the previous texts
            # are cleared and new texts are further used instead.
            # this means that if no texts are found, the old texts will stay
            # in place, which happens when no characters from a time step are
            # used.
            new_texts = []
            for token, probability in zip(self.tokens, timestep):
                if probability < min_probability:
                    continue
                for chars, text_probability in texts:
                    new_probability = text_probability * probability
                    new_texts.append((chars+token, new_probability))
            # if no characters were selected for this time step, and argument
            # ``or_best_char`` is ``True``, the single most likely character is
            # used.
            if len(new_texts) == 0 and or_best_char:
                max_i = np.argmax(timestep)
                probability = timestep[max_i]
                token = self.tokens[max_i]
                new_texts = [
                    (chars+token, text_probability*probability)
                    for chars, text_probability in texts
                ]
            if len(new_texts) > 0:
                texts = new_texts
        total_prob = 1.
        if normalize:
            total_prob = sum(d[1] for d in texts)
        result = {}
        for chars, probability in texts:
            # normalize each string into a final string to be returned.
            text = process_string_alnum(chars).upper()
            if text not in result:
                result[text] = 0.
            result[text] += probability / total_prob
        # if no texts were calculated, it means none of the characters in any
        # of the time steps had a sufficient probability and ``or_best_char``
        # was set to ``False``. Then, if ``or_best`` is ``True``, simply the
        # most probable text is used instead.
        if or_best and len(result) == 1 and '' in result:
            result = {self.text: 1.}
        if cache:
            self._cached[key] = result
        return result

    def similarity(
        self,
        others: typing.Union[Transcript, typing.List[Transcript]],
        algs_norm: NORM_ALGS_TYPE = NORM_ALGS_BEST_PERFORMING,
        algs_match: MATCH_ALGS_TYPE = MATCH_ALGS_BEST_PERFORMING,
        choose_best: bool = False,
        min_probability: typing.Optional[float] = None,
        or_best: bool = True,
        or_best_char: bool = False
    ) -> typing.Union[float, typing.List[float]]:
        """Calculate similarity scores over one or more other :class:`Transcript`\ s.

        These similarity scores are calculated by first normalizing texts taken
        from these transcripts, and then calculating a similarity score for
        each of them using the matching algorithms, and factors for how much
        they should count in the final result.

        Please also see the description in the function :meth:`probable_texts` of
        this class to read how the used probable texts are calculated.

        Args:
            others: The other transcripts to calculate a similarity score with.
            algs_norm: The algorithms to normalize the texts with before
                calculating similarity scores. The normalization algorithms
                are a representation of what should happen, which may include
                factors when combining algorithms. Possible algorithms are

                 - ``soundex`` for the Soundex algorithm [soundex]_
                 - ``nysiis`` for the NYSIIS algorithm [nysiis]_
                 - ``metaphone`` for the Metaphone algorithm [metaphone]_

                and these can be combined in the following ways

                 - one normalization algorithm can be given as::

                    algs_norm = 'soundex'

                 - multiple normalization algorithms can be given as::

                    algs_norm = ['soundex', 'nysiis']

                 - and factors can be given which do not need to add up to 1::

                    algs_norm = {'soundex': 0.1, 'nysiis': 0.5}

                The best performing combination of normalization algorithms and
                matching algorithms, according to the report, is used by
                default. For the normalization algorithms this is::

                    ['soundex', 'nysiis', 'metaphone']

            algs_match: The algorithms to use for calculating the actual
                similarities between strings after normalization using the
                algorithms in ``algs_norm``. The possible algorithms are

                 - ``hamming`` for the Hamming distance (as a similarity) [hamming]_
                 - ``levenshtein`` for the Levenshtein distance [levenshtein]_
                 - ``damerau`` for the Damerau-Levenshtein distance [damerau]_
                 - ``jaro`` for the Jaro similarity [jaro]_
                 - ``winkler`` for the Jaro-Winkler similarity [winkler]_

                and these can again be combined, in the way in which they can
                be similarly combined for the ``algs_norm`` argument as::

                    algs_match = 'hamming'
                    algs_match = ['hamming', 'jaro']
                    algs_match = {'hamming': 0.1, 'jaro': 0.5}

                and additionally as a dictionary taking the normalization
                algorithms into account as::

                    algs_match = {
                        'metaphone': {'hamming': 0.1, 'jaro': 0.3},
                        'nysiis': {'levenshtein': 0.1, 'jaro': 0.2}
                    }

                Here, again the best performing combinations are used by
                default, which for the matching algorithms is::

                    ['damerau', 'jaro', 'winkler']

            choose_best: If multiple probable texts for a single recording are
                used in the calculation of the similarity score, this argument
                determines if the total similarity score should be averaged, or
                if simply the higest similarity scores should be chosen.
                The default is ``False``, meaning that the final similarity score
                is averaged over all used probable texts.
            min_probability, or_best, or_best_char: Used in the
                :meth:`probable_texts` function, see the documentation there.

        Returns:
            A single similarity score if only a single transcript was given,
            or a list of similarity scores with one score for each of the given
            transcripts in order.

        .. [soundex] Wikipedia page on the Soundex algorithm https://en.wikipedia.org/wiki/Soundex
        .. [nysiis] Wikipedia page on the NYSISS algorithm https://en.wikipedia.org/wiki/New_York_State_Identification_and_Intelligence_System
        .. [metaphone] Wikipedia page on the Metaphone algorithm https://en.wikipedia.org/wiki/Metaphone
        .. [hamming] Wikipedia page on the Hamming distance https://en.wikipedia.org/wiki/Hamming_distance
        .. [levenshtein] Wikipedia page on the Levenshtein distance https://en.wikipedia.org/wiki/Levenshtein_distance
        .. [damerau] Wikipedia page on the Damerau-Levenshtein distance https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
        .. [jaro] Wikipedia page on the Jaro similarity https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance#Jaro_similarity
        .. [winkler] Wikipedia page on the Jaro-Winkler similarity https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance#Jaro%E2%80%93Winkler_similarity
        """
        single = type(others) is Transcript
        others = ensure_list(others)
        algs_norm, algs_match = ensure_algs_dict(algs_norm, algs_match)
        algs_match = {k: dicts_to_tuples(v) for k, v in algs_match.items()}
        # select the texts to be used, with either a minimum probability, or
        # not.
        if min_probability is not None:
            texts = self.probable_texts(min_probability, or_best=or_best,
                                        or_best_char=or_best_char)
        else:
            texts = {self.text: 1}
        results = []
        for other in others:
            if min_probability is not None:
                other_texts = other.probable_texts(min_probability, or_best=or_best,
                                                   or_best_char=or_best_char)
            else:
                other_texts = {other.text: 1}
            result = []
            for text, probability in texts.items():
                for other_text, other_probability in other_texts.items():
                    # for each of the probable texts in this Transcript and
                    # the other Transcript, go through the normalization and
                    # matching algorithms and calculate a score and multiply
                    # this score with the combined factor of the normalization
                    # and matching algorithm factors.
                    factor = probability * other_probability
                    inner_result = []
                    for alg_norm, value in algs_norm.items():
                        this_norm = normalize(text, alg_norm)
                        other_norm = normalize(other_text, alg_norm)
                        norm_result = sum(match(
                            this_norm,
                            other_norm,
                            algs_match[alg_norm]
                        ))
                        inner_result.append(value*norm_result)
                    inner_mean = sum(inner_result) / len(inner_result)
                    if not choose_best:
                        inner_mean *= factor
                    result.append(inner_mean)
            # take the maximum similarity score or calculate an average.
            if choose_best:
                result = max(result)
            else:
                result = sum(result) / len(result)
            results.append(result)
        if single:
            assert len(results) == 1
            results = results[0]
        return results


class Recording:
    """Representation of an audio recording and various processing tasks.

    Args:
        filepath: The path to the audio file.
        transcriptor: The :class:`speechmatching.model.Transcriptor` to use for the transcribing the
            recording. This is optional. If not given, a global one is
            created and used.
        preload: Whether the audio file should be processed upon
            initialization of this class. This may significantly increase
            the time required for initializing the class, but prevents this
            processing time from happening later. ``True`` by default.
        identifier: The identifier of the :class:`Recording` instance. This
            identifier can be used in the :class:`Group` the recording is in to
            allow for looking up the :class:`Recording`. If not given, this will
            be set to ``filepath``.
        raw_output_filepath: The filepath to which the raw output of the
            audio file given by the :class:`speechmatching.model.Transcriptor` will be written. If not
            given, this is set to the audio ``filepath`` with ``_raw_output.txt``
            appended to it.
    """

    # The global transcriptor. Since it is expensive to run an transcript due
    # to the memory taken, a transcriptor can be made 'global' by putting it in
    # this constant. It can be then be used by every :class:`Recording`
    # instance.
    GLOBAL_TRANSCRIPTOR = None

    def __init__(
        self,
        filepath: str,
        transcriptor: typing.Optional[Transcriptor] = None,
        preload: bool = True,
        identifier: typing.Optional[str] = None,
        raw_output_filepath: typing.Optional[str] = None
    ):
        self._filepath = filepath
        self._transcriptor = transcriptor or self.get_global_transcriptor()
        self._identifier = identifier or self._filepath
        self._raw_output_filepath = raw_output_filepath \
            or (self._filepath + '_raw_output.txt')
        if preload:
            _ = self.transcript

    @classmethod
    def get_global_transcriptor(cls) -> Transcriptor:
        """Create and/or return the created global :class:`speechmatching.model.Transcriptor`.

        Returns:
            The global :class:`speechmatching.model.Transcriptor`."""
        if cls.GLOBAL_TRANSCRIPTOR is None:
            cls.GLOBAL_TRANSCRIPTOR = Transcriptor()
        return cls.GLOBAL_TRANSCRIPTOR

    @property
    def filepath(self) -> str:
        """The filepath of the audio file."""
        return self._filepath

    @property
    def identifier(self) -> str:
        """The identifier of the current :class:`Recording` instance."""
        return self._identifier

    def similarity(
        self,
        others: typing.Union[Recording, typing.List[Recording]],
        *args,
        **kwargs
    ) -> typing.Union[float, typing.List[float]]:
        """Calculate similarity scores between this and other :class:`Recording`\ s.

        Examples:
            >>> rec1 = Recording('audio1.wav')
            >>> rec2 = Recording('audio2.wav')
            >>> rec1.similarity(rec2, algs_norm='metaphone', algs_match='jaro')
            0.85
            >>> rec3 = Recording('audio3.wav')
            >>> rec4 = Recording('audio4.wav')
            >>> rec1.similarity([rec2, rec3, rec4], algs_norm='metaphone', algs_match='jaro')
            [0.85, 0.3, 0.91]

        Args:
            others: The other recording to calculate a similarity to, or a list
                of multiple other recordings.
            *args, **kwargs: Arguments for the :meth:`Transcript.similarity`
                function of the :class:`Transcript` instance for this current
                :class:`Recording`.

        Returns:
            A single similarity score if one :class:`Recording` was given, or a list
            of scores if multiple :class:`Recording`\ s were given.
        """
        if type(others) is list:
            others = [other.transcript for other in others]
        else:
            others = others.transcript
        return self.transcript.similarity(others, *args, **kwargs)

    def match(
        self,
        groups: typing.List[Group],
        size: int = sys.maxsize,
        use_min_group_size: bool = False,
        algs_norm: NORM_ALGS_TYPE = NORM_ALGS_BEST_PERFORMING,
        algs_match: MATCH_ALGS_TYPE = MATCH_ALGS_BEST_PERFORMING,
        return_indecision: bool = True,
        *args,
        **kwargs
    ) -> typing.Union[None, Group, typing.List[Group]]:
        """Match the current :class:`Recording` with multiple :class:`Group`\ s.

        Matching of one recording with multiple groups will allow for a single
        group, multiple groups, or no groups to be found to match the
        recording depending on the given arguments, see the descriptions of the
        parameters for more information on this.

        The normalization and matching algorithms used here are by default set
        to the values that were found to work best according to the thesis for
        which this software was written.

        Examples:
            >>> group1 = Group(identifier="Group1", recordings=[rec2])
            >>> group2 = Group(identifier="Group2", recordings=[rec3, rec4])
            >>> rec1.match([group1, group2], size=1)
            Group(identifier='Group2', ...)

        Args:
            groups: The groups to match this recording against.
            size: The number of recordings from each group to use for matching.
                If a group has fewer recordings than this ``size``, then
                argument ``use_min_group_size`` decides how many recordings of
                each group will be used.
            use_min_group_size: If set to ``True``, then if one of the groups has
                fewer than ``size`` recordings, ``size`` is readjusted to the
                size of this group so an equal number of recordings from each
                group is used.
            algs_norm: See the documentation for the
                :meth:`Transcript.similarity` function. Here, the default value
                is set to::

                    ['soundex', 'nysiis', 'metaphone']

            algs_match: See the documentation for the
                :meth:`Transcript.similarity` function. Here, the default value
                is set to::

                    ['damerau', 'jaro', 'winkler']

            return_indecision: Whether to return ``None`` if there are multiple
                groups that match the recording with the same similarity. When
                this happens, a decision cannot be made for a single group. If
                this argument is set to ``False``, all matched groups are
                returned instead.
            *args, **kwargs: Arguments for the :meth:`similarity` function.

        Returns:
            One of the given groups is returned if this group was found to be
            a best match to the recording. Multiple groups are returned if
            argument ``return_indecision`` is ``False``, and multiple groups match
            the recording equally well. ``None`` is returned if multiple groups
            match equally well and ``return_indecision`` is set to `True`.
        """
        # set the maximum number of recordings to take from each group.
        if use_min_group_size:
            min_group_size = min(len(group) for group in groups)
            if min_group_size < size:
                size = min_group_size
        candidates = []
        mapping = {}
        j = -1
        # take a certain number of recordings from each group, and add them to
        # a list. which group a recording belongs to is a tracked by storing
        # the index of its group in a dictionary.
        for i, group in enumerate(groups):
            for j, recording in enumerate(
                group.sample(k=min(size, len(group))), start=j+1
            ):
                candidates.append(recording)
                mapping[j] = i
        similarities = self.similarity(
            candidates,
            algs_norm=algs_norm,
            algs_match=algs_match,
            *args,
            **kwargs
        )
        similarities = sorted(
            enumerate(similarities),
            key=lambda x: x[1], reverse=True
        )
        matches = []
        # get the groups that have the highest similarity score.
        for i, score in similarities:
            if score == similarities[0][1]:
                match = groups[mapping[i]]
                if match not in matches:
                    matches.append(match)
        assert len(matches) > 0
        # decide what to do if there are multiple equally well matching groups.
        if return_indecision:
            if len(matches) > 1:
                return None
            else:
                return matches[0]
        return matches

    @functools.cached_property
    def transcript(self) -> Transcript:
        """Get the :class:`Transcript` of this recording.

        The recording is transcribed with the :class:`speechmatching.model.Transcriptor`, which may be a global
        one, as detailed earlier in this class. If the file in which the raw
        output would be stored exists, it is loaded and no new transcript is
        created. If it does not exist, it is created after transcribing.

        Returns:
            The transcript of the :class:`Recording`.
        """
        if self._raw_output_filepath is not None \
            and os.path.isfile(self._raw_output_filepath):
            with open(self._raw_output_filepath) as f:
                return Transcript(raw=f.read())
        raw_output = self._transcriptor.transcribe(self._filepath)
        with open(self._raw_output_filepath, 'w') as f:
            f.write(raw_output)
        return Transcript(raw=raw_output)


class Group:
    """A group to hold :class:`Recording`\ s.

    Args:
        identifier: The identifier of the group, which should be unique. It is
            not required.
        labels: The labels of the group. This could for example be names or
            translations::

                {
                    "English": "Three",
                    "Dagbani": "Ata"
                }

            An empty dictionary is used if nothing is given.
        recordings: The list :class:`Recording`\ s to initially load. If
            nothing is given, the group is initialized empty.
    """

    def __init__(
        self,
        identifier: typing.Optional[str] = None,
        labels: typing.Optional[typing.Dict[str, str]] = None,
        recordings: typing.Optional[typing.List[Recording]] = None,
    ):
        self._identifier = identifier
        self._labels = labels or dict()
        self._recordings = {}
        self._recordings_list = []
        if recordings is not None:
            for recording in recordings:
                self.add(recording)

    def __len__(self) -> int:
        """The number of :class:`Recording`\ s in the :class:`Group`\ ."""
        return len(self._recordings_list)

    def add(self, recording: Recording) -> bool:
        """Add a :class:`Recording`\ .

        Args:
            recording: The :class:`Recording` to add under the identifier set in the
                :class:`Recording` itself.

        Returns:
            ``True`` if the recording was successfully added, or ``False`` if
            the identifier from the given recording is already in this group.
        """
        if recording.identifier in self._recordings:
            return False
        self._recordings[recording.identifier] = recording
        self._recordings_list.append(recording)
        return True

    def remove(self, recording: typing.Union[Recording, str]):
        """Remove a recording from the :class:`Group`\ .

        Args:
            recording: The identifier to the :class:`Recording`, or the :class:`Recording`
                itself that should be removed from this :class:`Group`\ .

        Raises:
            ValueError: If the recording does not exist in this group.
        """
        if type(recording) is str:
            recording = self._recordings[recording]
        if recording not in self._recordings_list:
            raise ValueError('Recording does not exist.')
        self._recordings_list.remove(recording)
        del self._recordings[recording.identifier]

    def set_label(self, k: str, v: str):
        """Set or replace a label.

        Args:
            k: The key of the label.
            v: The value of the label.
        """
        self._labels[k] = v

    @property
    def identifier(self) -> typing.Optional[str]:
        """The identifier of the :class:`Group`\ ."""
        return self._identifier

    def label(self, k: str) -> typing.Optional[str]:
        """Get a label from the group.

        Args:
            k: The key of the label.

        Returns:
            The label under key ``k``.
        """
        return self._labels.get(k)

    def recording(self, identifier: str) -> Recording:
        """Get the :class:`Recording` under a certain identifier.

        Args:
            identifier: The identifier of the :class:`Recording`\ .

        Returns:
            The :class:`Recording` under label ``identifier``.
        """
        return self._recordings[identifier]

    def recordings(self) -> typing.List[Recording]:
        """Return a list of all :class:`Recording`\ s in this :class:`Group`\ ."""
        return list(self._recordings_list)

    def sample(self, k: int = 1) -> typing.List[Recording]:
        """A number of randomly picked unique recordings from the :class:`Group`\ .

        Args:
            k: The number of :class:`Recording`\ s to get. The default value is
                1.

        Returns:
            The list of ``k`` randomly picked unique :class:`Recording`\ s.
         """
        return random.sample(self._recordings_list, k=k)

    def random(self) -> Recording:
        """Return a single random :class:`Recording`\ ."""
        return self.sample(k=1)[0]

    def group(
        self,
        size: typing.Optional[int] = None,
        identifier: typing.Optional[str] = None
    ) -> Group:
        """Create a new :class:`Group` from this :class:`Group`\ .

        Args:
            size: The number of random :class:`Recording`\ s from this
                :class:`Group` to use. This is by default set to the number of
                :class:`Recording`\ s currently in this :class:`Group`.
            identifier: The identifier for the new :class:`Group`\ . This is by
                default set to the identifier of the current :class:`Group`
                with ``_sub`` appended to it.

        Returns:
            The new :class:`Group` of given ``size`` and with given
            ``identifier``.
        """
        identifier = identifier or self._identifier
        if identifier is not None:
            identifier += '_sub'
        if size is None:
            size = len(self)
        return Group(
            identifier=identifier,
            labels=self._labels,
            recordings=self.sample(k=size)
        )


def load_directory(
    directory: str,
    transcriptor: typing.Optional[Transcriptor] = None,
    identifier: typing.Optional[str] = None,
    labels: typing.Optional[typing.Dict[str, str]] = None,
    return_empty: bool = False,
    verbose: bool = True
) -> typing.Optional[Group]:
    """Create a :class:`Group` with all audio files in a directory.

    Any audio files ending with ``_processed.wav`` in the directory will be
    ignored, as this is the filename ending used for audio files that have been
    converted into the format used by the ``acoustic`` binary for transcribing.

    Args:
        directory: The directory from which load the audio files.
        transcriptor: The :class:`speechmatching.model.Transcriptor` to use for
            transcribing. If not given, a global one will be used.
        identifier: The identifier of the new :class:`Group`. If not set, the
            name of the directory will be used instead.
        labels: A dictionary of the labels to set.
        return_empty: If set to ``True``, an empty :class:`Group` is created if
            the directory contains no audio files. Default is ``False``.
        verbose: Whether to output the progress of loading files. Default is
            ``True``.

    Returns:
        The :class:`Group` with loaded audio files, or ``None`` if no audio
        files were found, and ``return_empty`` is ``False``.
    """
    recordings = []
    filenames = os.listdir(directory)
    for i, filename in enumerate(filenames):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath) \
            or filepath.endswith('_processed.wav') \
            or filepath.endswith('_raw_output.txt'):
            continue
        if magic.from_file(filepath, mime=True).split('/', 1)[0] not in ('audio', 'video'):
            continue
        if verbose:
            print('Loading file {}/{}: {}'.format(i, len(filenames), filepath))
        recordings.append(Recording(filepath, transcriptor=transcriptor))
    if return_empty or len(recordings) > 0:
        return Group(
            identifier=identifier or directory,
            labels=labels,
            recordings=recordings
        )
    return None


def load_directory_groups(
    directory: str,
    transcriptor: typing.Optional[Transcriptor] = None,
    return_empty: bool = False
) -> typing.Dict[str, Group]:
    """Load multiple :class:`Group`\ s of :class:`Recording`\ s from a directory.

    The given ``directory`` should contain multiple directories which each
    containing the audio files for a :class:`Group`\ . If a directory does not
    contain audio files, it may be ignored depending on the ``return_empty``
    argument.

    If a ``<directory>_metadata.json`` file is present in the directory next to
    the directory containing :class:`Recording`\ s for the :class:`Group`\ ,
    the identifier and labels from this JSON file will be adopted for the new
    :class:`Group`\ , else default values documented in the arguments of this
    function will be used.

    The ``<directory>_metadata.json`` is of format JSON and can hold one of the
    keys ``identifier`` and ``label``, with ``label`` being a dictionary with
    strings as keys and values. An example is::

        {
            "identifier": "mygroup",
            "labels": {
                "en": "my group",
                "nl": "mijn groep"
            }
        }

    Args:
        directory: The directory from which directories of audio files are
            loaded into :class:`Group`\ s.
        transcriptor: The :class:`speechmatching.model.Transcriptor` to use for
            the processing the audio files. If not given, a global one will be
            used.
        return_empty: Whether to create an empty :class:`Group` if a subdirectory
            does not contain any audio files.

    Returns:
        A dictionary of the created :class:`Group`\ s with their identifier as key,
        and the created :class:`Group` as value.
    """
    groups = {}
    for subdir in os.listdir(directory):
        identifier = subdir
        labels = None
        metadata_file = subdir+'_metadata.json'
        if os.path.isfile(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                identifier = metadata.get('identifier') or identifier
                labels = metadata.get('labels')
        group = load_directory(
            os.path.join(directory, subdir),
            transcriptor=transcriptor,
            identifier=identifier,
            labels=labels,
            return_empty=return_empty
        )
        if group is not None:
            groups[identifier] = group
    return groups

