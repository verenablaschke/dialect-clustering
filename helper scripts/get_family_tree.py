# Get the hierarchical language family information from Glottolog
# for the Sound Comparisons doculects.

import csv

DOCULECT_INFO = 'data/soundcomparisons/glottolog/glottolog_codes.csv'
GLOTTOLOG_INFO = 'data/glottolog/languoid.csv'

lang2parent = {}
lang2name = {}
with open(GLOTTOLOG_INFO, encoding='utf8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        lang2name[row['id']] = row['name']
        lang2parent[row['id']] = row['parent_id']


with open(DOCULECT_INFO, encoding='utf8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        lang = row['Filename']
        code = row['Code']
        print(lang)
        try:
            print(code, lang2name[code])
            while True:
                code = lang2parent[code]
                print(code, lang2name[code])
        except KeyError:
            pass
        print()

