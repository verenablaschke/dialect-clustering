import os
import pandas as pd

DOCULECTS = ['High German (Biel)', 'High German (Bodensee)',
             'High German (Graubuenden)', 'High German (Herrlisheim)',
             # 'High German (North Alsace)',
             'High German (Ortisei)',
             'High German (Tuebingen)', 'High German (Walser)',
             'Central German (Cologne)', 'Central German (Honigberg)',
             'Central German (Luxembourg)', 'Central German (Murrhardt)',
             'German',
             # Achterhoeks is spoken in NL, but it's Low German (Glottolog)
             'Low German (Achterhoek)', 'Low German (Bargstedt)',
             'Dutch', 'Belgian Dutch',
             # Limburg here refers to the Dutch province
             'Dutch (Antwerp)', 'Dutch (Limburg)', 'Dutch (Ostend)',
             'West Frisian (Grou)',
             # including Yiddish just out of curiosity
             'Yiddish (New York)'
             ]


def get_samples(dir_bdpa='data/bdpa',
                dir_soundcomparisons='data/soundcomparisons',
                doculects=DOCULECTS):
    entries = get_samples_bdpa(dir_bdpa)
    entries = get_samples_soundcomparisons(dir_soundcomparisons, entries)
    return entries


def get_samples_bdpa(directory, doculects=DOCULECTS):
    """"Extracts phonetic transcriptions from multiple BDPA MSA files.

    Args:
        directory (str): The directory containing MSA files from the BDPA.

    Keyword args:
        doculects (list(str)): The names of the doculects for which information
                               should be extracted. (default: DOCULECTS)

    Returns:
        dict(str -> dict(str -> str)): A mapping from concepts to doculects to
                                       to entries.
    """
    entries = {}
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.msa'):
                concept, subentries = parse_file_bdpa(os.path.join(root, f),
                                                      doculects)
                entries[concept] = subentries
    return entries


def parse_file_bdpa(filename, doculects):
    """Extracts phonetic transcriptions from a BDPA MSA file.

    Args:
        filename (str): The MSA file.
        doculects (list(str)): The names of the relevant doculects.

    Returns:
        str: The English cognate from this file.
        dict(str -> str): A dictionary from doculects to word entries.
    """
    with open(filename, encoding='utf8') as f:
        lines = [line.replace('.', '').replace('\n', '').replace('-', '')
                 for line in f.readlines()]
        # Line 0: language family
        # Line 1: term (in EN)
        concept = lines[1].replace('"', '')
        if concept.startswith('to '):
            concept = concept[3:]
        # Other lines: language_name......\tsegment_1\tsegment_2\tsegment_n
        entries = {}
        for line in lines[2:]:
            cells = line.split('\t')
            doculect = cells[0]
            if doculect in doculects:
                entries[doculect] = (''.join(cells[1:])
                                     .replace('(', '').replace(')', ''))
        for doculect in doculects:
            try:
                entries[doculect]
            except KeyError:
                print('{} has no entry for {}.'.format(doculect, concept))
    return concept, entries


def get_samples_soundcomparisons(directory, entries):
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.csv'):
                entries = parse_file_soundcomparisons(os.path.join(root, f),
                                                      entries)
    return entries


def parse_file_soundcomparisons(filename, entries):
    df = pd.read_csv(filename, encoding='utf8')
    if ('/') in filename:
        filename = filename.split('/')[-1]
    if ('\\') in filename:
        filename = filename.split('\\')[-1]
    doculect = filename[:-4]
    concepts = df['WordModernName1'].values
    words = df['Phonetic'].values
    noncognate = df['NotCognateWithMainWordInThisFamily2'].values
    for i, concept in enumerate(concepts):
        try:
            word = str(words[i]).strip().replace('.', '')
            if len(word) == 0:
                print('{} has an empty entry for {} (skipped)'
                      .format(doculect, concept))
                continue
            if noncognate[i] > 0:
                print('{} has a non-cognate entry for {} ({}) (skipped)'
                      .format(doculect, concept, word))
                continue
            entries[concept][doculect] = word
        except KeyError:
            entries[concept] = {doculect: word}
            print('{} has an entry for {}, '
                  'which did not appear in the BDPA files (added)'
                  .format(doculect, concept))
    for concept in entries:
        try:
            entries[concept][doculect]
        except KeyError:
            print('{} has no entry for {}.'.format(doculect, concept))
    return entries
