# Retrieve IPA transcriptions by concept from the Sound Comparisons files.

import logging
import numpy as np
import os
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_samples(dir_cwg='data/soundcomparisons/cwg/',
                dir_additional='data/soundcomparisons/additional/',):
    entries = {}
    entries, doculects_cwg = get_samples_soundcomparisons(dir_cwg, entries)
    entries, doculects_add = get_samples_soundcomparisons(dir_additional,
                                                          entries)
    return entries, sorted(doculects_cwg), sorted(doculects_add)


def clean_transcription(word):
    # Removing blank space characters is necessary because sometimes the
    # 'no voicing' diacritic is combined with a (superfluous) blank space
    # instead of an IPA character.
    to_remove = ['.', ' ', '(', ')', '[', ']', 'ˈ', 'ˌ']
    word = str(word).strip()
    for c in to_remove:
        word = word.replace(c, '')
    # Change LATIN SMALL LETTER C + COMBINING CEDILLA
    # to LATIN SMALL LETTER C WITH CEDILLA so LingPy deals with it properly.
    word = word.replace('ç', 'ç')
    word = word.replace('g', 'ɡ')
    affricates = {'ts': 't͡s', 'dz': 'd͡z', 'tʃ': 't͡ʃ', 'dʒ': 'd͡ʒ',
                  'pf': 'p͡f', 'kx': 'k͡x'}
    for k, v in affricates.items():
        word = word.replace(k, v)
    return word


def get_samples_soundcomparisons(directory, entries):
    doculects = set()
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.csv'):
                entries, doculect = parse_file(
                    os.path.join(root, f), entries)
                doculects.update([doculect])
    return entries, doculects


def parse_file(filename, entries):
    df = pd.read_csv(filename, encoding='utf8')
    if ('/') in filename:
        filename = filename.split('/')[-1]
    if ('\\') in filename:
        filename = filename.split('\\')[-1]
    doculect = filename[:-4]
    # TODO del
    print(filename, df.columns)
    df['Phonetic'].replace('nan', np.nan, inplace=True)
    df.dropna(subset=['Phonetic'], inplace=True)
    concepts = df['WordModernName1'].values
    words = df['Phonetic'].values
    noncognate = df['NotCognateWithMainWordInThisFamily'].values
    for concept, w, n in zip(concepts, words, noncognate):
        word = clean_transcription(w)
        # if word == 'Array':
        #     # Erroneous entry in Veenkolonien.csv.
        #     continue
        if len(word) == 0:
            logger.info('{} has an empty entry for {} (skipped)'
                        .format(doculect, concept))
            continue
        if n > 0:
            logger.info('{} has a non-cognate entry for {} ({}) '
                        '(skipped)'.format(doculect, concept, word))
            continue
        try:
            entries[concept][doculect] = word
        except KeyError:
            entries[concept] = {doculect: word}
    for concept in entries:
        try:
            entries[concept][doculect]
        except KeyError:
            logger.info('{} has no entry for {}.'.format(doculect, concept))
    return entries, doculect
