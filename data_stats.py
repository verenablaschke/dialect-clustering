from collections import Counter
from read_data import clean_transcription
import numpy as np
import os
import pandas as pd

dir_cwg = 'data/soundcomparisons/cwg/'

docs = {}
concepts = Counter()
for root, dirs, files in os.walk(dir_cwg):
    for f in files:
        if f.endswith('.csv') and 'ProtoGermanic' not in f:
            df = pd.read_csv(dir_cwg + f, encoding='utf8')
            # Remove faulty/empty concepts.
            df['Phonetic'] = df['Phonetic'].apply(clean_transcription)
            df['Phonetic'].replace('nan', np.nan, inplace=True)
            df.dropna(subset=['Phonetic'], inplace=True)
            entries = df['Phonetic'].values
            words = df['WordId'].values
            if ('/') in f:
                filename = f.split('/')[-1]
            if ('\\') in f:
                filename = f.split('\\')[-1]
            doculect = f[:-4]
            docs[doculect] = len(entries)
            concepts.update(words)

print(docs)
print()
print("number of doculects", len(docs))
print()
docs = np.fromiter(docs.values(), dtype=np.int8)
print("min", np.amin(docs))
print("max", np.amax(docs))
print("mean", np.mean(docs))
print("std", np.std(docs))
print()
print()

print(concepts.most_common())
print()
print("number of concepts", len(concepts))
print()
concepts = np.fromiter(concepts.values(), dtype=np.int8)
print("min", np.amin(concepts))
print("max", np.amax(concepts))
print("mean", np.mean(concepts))
print("std", np.std(concepts))
