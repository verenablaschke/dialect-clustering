import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_samples(dir_cwg='data/soundcomparisons/cwg/',
                dir_additional='data/soundcomparisons/additional/',):
    entries = {}
    id2concept = construct_id2concept(dir_cwg)
    entries, doculects_cwg = get_samples_soundcomparisons(dir_cwg,
                                                          entries, id2concept)
    entries, doculects_add = get_samples_soundcomparisons(dir_additional,
                                                          entries, id2concept)
    return entries, sorted(doculects_cwg), sorted(doculects_add)


def clean_transcription(word):
    # Removing blank space characters is necessary because sometimes the
    # 'no voicing' diacritic is combined with a (superfluous) blank space
    # instead of an IPA character.
    return str(word).strip().replace('.', '').replace(' ', '') \
                    .replace('(', '').replace(')', '') \
                    .replace('[', '').replace(']', '') \
                    .replace('ˈ', '').replace('̩', '') \
                    .replace('ʦ', 't͡s').replace('t͡ʃ', 'ʧ')  # TODO check
    # TODO more?


def get_samples_soundcomparisons(directory, entries, id2concept):
    doculects = set()
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.csv'):
                entries, doculect = parse_file(
                    os.path.join(root, f), entries, id2concept)
                doculects.update([doculect])
    return entries, doculects


def construct_id2concept(directory):
    df = pd.read_csv(directory + 'ProtoGermanic.csv', encoding='utf8')
    concepts = df['WordModernName1'].values
    ids = df['WordId'].values
    return {i: c for i, c in zip(ids, concepts)}


def parse_file(filename, entries, id2concept):
    df = pd.read_csv(filename, encoding='utf8')
    if ('/') in filename:
        filename = filename.split('/')[-1]
    if ('\\') in filename:
        filename = filename.split('\\')[-1]
    doculect = filename[:-4]
    ids = df['WordId'].values
    words = df['Phonetic'].values
    noncognate = df['NotCognateWithMainWordInThisFamily2'].values
    for i, w, n in zip(ids, words, noncognate):
        concept = id2concept[i]
        # TODO check if this is actually correct all cases
        word = clean_transcription(w)
        if word == 'Array':
            # Erroneous entry in Veenkolonien.csv.
            continue
        if len(word) == 0:
            logger.info('{} has an empty entry for {}/{} (skipped)'
                        .format(doculect, i, concept))
            continue
        if n > 0:
            logger.info('{} has a non-cognate entry for {}/{} ({}) '
                        '(skipped)'.format(doculect, i, concept, word))
            continue
        entries[concept][doculect] = word
    for concept in entries:
        try:
            entries[concept][doculect]
        except KeyError:
            logger.info('{} has no entry for {}.'.format(doculect, concept))
    return entries, doculect
