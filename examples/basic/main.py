"""Basic example application that uses directories of recordings."""

import hashlib
import os
import tempfile
import zipfile

import requests
import speechmatching

from speechmatching import Transcriptor, Group, Recording
from speechmatching.recording import load_directory_groups

from speechmatching.match import ensure_norm_algs_dict, ensure_algs_dict


def download_recordings():
    """Download recordings of word in Dagbani [dagbani]_.

    .. [dagbani] F. Saa-Dittoh and Tingoli and Nyankpala Communities of the Northern
        Region of Ghana, "Recordings of spoken dagbani for the letters zero to
        ten and the words yes and no", dag, Zenodo, 2024.
        doi: 10.5281/ZENODO.13284005. [Online]. Available:
        http://dx.doi.org/10.5281/ZENODO.13284005.
    """
    response = requests.get('https://zenodo.org/api/records/13284005/files/recordings.zip')
    assert response.status_code == 200
    response = response.json()
    content = response['links']['content']
    checksum = response['checksum']
    alg, value = checksum.split(':')
    hash_calculator = getattr(hashlib, alg)()
    with tempfile.TemporaryFile(mode='wb+', suffix='.zip') as tf:
        response = requests.get(content, stream=True)
        assert response.status_code == 200
        for chunk in response.iter_content(chunk_size=1024**2):
            if chunk:
                tf.write(chunk)
                hash_calculator.update(chunk)
        tf.flush()
        assert hash_calculator.hexdigest() == value
        with zipfile.ZipFile(tf, mode='r') as zf:
            zf.extractall()


def basic1():
    print('# Basic example 1')

    print('Loading a recording \'Ata\' (English: \'Three\')')
    speech1 = Recording('recordings/Three - Ata/recording-0b4c4df8-311c-4b46-b9f3-6158211b3887.3gp')
    print('  ', speech1)

    print('We get the transcript.')
    # get the transcript
    transcript = speech1.transcript
    print('  ', transcript)
    print('The most likely transcript is', transcript.text)
    print('The top most likely transcripts with minimum character probability of 0.25 is:')
    most_probable_texts = transcript.probable_texts(min_probability=0.25)
    for k, v in sorted(most_probable_texts.items(), key=lambda x: x[1], reverse=True):
        print(' - {}, {}'.format(k, v))

    print('We load a second recording, also for \'Ata\'.')
    # calculate similarity scores to other recordings
    speech2 = Recording('recordings/Three - Ata/recording-0e8b13fa-4f85-4841-967c-1c6f4e9bb416.3gp')
    print('  ', speech2)
    print('The similarity scores between the recordings is',
          speech2.similarity(speech1))
    print('Which can be also be retrieved using the transcripts of both',
          speech2.transcript.similarity(speech1.transcript))
    print('Similarity score', speech2.similarity(speech1))
    print('')


def basic2():
    print('# Basic example 2')

    print('Loading three groups of different words, with each at least 2 recordings.')
    groups = [
        Group(
            identifier='ata',
            recordings=[
                Recording('recordings/Three - Ata/recording-0b4c4df8-311c-4b46-b9f3-6158211b3887.3gp'),
                Recording('recordings/Three - Ata/recording-0e8b13fa-4f85-4841-967c-1c6f4e9bb416.3gp'),
                Recording('recordings/Three - Ata/recording-27f923f2-aab5-467e-b554-ba58282311c6.3gp')
            ]
        ),
        Group(
            identifier='pia',
            recordings=[
                Recording('recordings/Ten - Pia/recording-0ee929a8-9534-4f37-adbe-891f4fca59de.3gp'),
                Recording('recordings/Ten - Pia/recording-a46bc368-9c4f-4893-843a-47e67997572d.3gp')
            ]
        ),
        Group(
            identifier='anahi',
            recordings=[
                Recording('recordings/Four - Anahi/recording-5bb02c2c-1ca0-4405-ad39-7271c7688a38.3gp'),
                Recording('recordings/Four - Anahi/recording-573c5043-495b-44ce-ab7c-eea2648d0093.3gp'),
                Recording('recordings/Four - Anahi/recording-66420e57-8ced-4873-a2f3-03778b9d4b09.3gp'),
                Recording('recordings/Four - Anahi/recording-c6ab73ea-c783-4465-aa73-25b7487069a8.3gp')
            ]
        )
    ]
    print('  ', groups)

    print('Load a second \'unknown\' recording (but we know it\'s \'Anahi\').')
    unknown = Recording('recordings/Four - Anahi/recording-eacb6533-42d3-4fcf-a9b3-3f608a6326b6.3gp')
    print('  ', unknown)
    match = unknown.match(groups)
    print('The best matching group is', match.identifier)
    print('')


def basic3():
    print('# Basic example 3')
    print('Load all recordings from directory \'recording\' into groups using '
          'their directories.')
    groups = load_directory_groups('recordings')

    for identifier, group in groups.items():
        print(' - group \'{}\' with {} recordings.'.format(identifier, len(group)))

    print('Take a test group, \'Ata\'.')
    test_group = groups['Three - Ata']
    print('  ', test_group)

    print('For 20 random words from \'{}\' the best match is:'.format(test_group.identifier))
    for recording in test_group.sample(k=20):
        # remove the chosen recording from the test group
        test_group.remove(recording)
        # match the chosen recordings with all groups (and all recordings,
        # except the removed one)
        result = recording.match(
            # the list of groups, from the earlier loaded dictionary
            list(groups.values()),
            # use from each group a number of recording equal to the number of
            # recording in the group with the smallest number
            use_min_group_size=True,
            # return None if no decision can be made (instead of returning all
            # equally well matching groups)
            return_indecision=True
        )
        # get the matching groups identifier, or set that no decision was taken
        if result:
            result = result.identifier
        else:
            result = '*no decision*'
        print('\t\'{}\' found for recording \'{}\''
              .format(result, recording.identifier.rsplit('/', 1)[1]))
        # add the temporarily deleted recording back to the group
        test_group.add(recording)

if __name__ == '__main__':
    # download the recordings if not downloaded yet
    if not os.path.isdir('recordings'):
        print('Downloading recordings.')
        download_recordings()
    basic1()
    basic2()
    basic3()

