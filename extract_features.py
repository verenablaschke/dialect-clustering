import os
import panphon
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
from matplotlib import pyplot as plt

DATA_DIR = 'data'
VARIANTS = ['High German (Biel)', 'High German (Bodensee)',
            'High German (Graubuenden)', 'High German (Herrlisheim)',
            'High German (North Alsace)', 'High German (Ortisei)',
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
            # 'Proto-Germanic',
            # 'Yiddish (New York)'
            ]
ft = panphon.FeatureTable()


def parse_file(filename):
    with open(filename, encoding='utf8') as f:
        lines = [line.replace('.', '').replace('\n', '')
                 for line in f.readlines()]
        # Line 0: language family
        # Line 1: term (in EN)
        concept = lines[1].replace('"', '')
        if concept == 'tear':
            # 'tear' is divided into two files, each with a different concept
            # set that only covers about half the doculects.
            return None
        # Other lines: language_name......\tphonetic_representation
        entries = {}
        for line in lines[2:]:
            cells = line.split('\t')
            variant = cells[0].replace('.', '')
            if variant in VARIANTS:
                entries[variant] = tokens2vec(cells[1:])
        for variant in VARIANTS:
            try:
                entries[variant]
            except KeyError:
                # TODO: Variant is missing from the file.
                print('{} has no entry for {}.'.format(variant, concept))
                entries[variant] = tokens2vec(['-' for _ in range(len(cells) - 1)])
    return entries


def tokens2vec(tokens):
    segs = []
    for token in tokens:
        # TODO: diphthongs!
        try:
            segs += ft.fts(token).numeric()
        except AttributeError:
            # Couldn't convert the IPA token,
            # probably because it's an insertion/deletion dummy token.
            segs += null_segment()
    return segs


def null_segment():
    # TODO better approach?
    return [0 for _ in range(22)]


entries = {}
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.msa'):
            subentries = parse_file(os.path.join(root, f))
            if subentries is None:
                continue
            for e in subentries:
                try:
                    entries[e] += subentries[e]
                except KeyError:
                    entries[e] = subentries[e]

n_samples = len(VARIANTS)
n_features = len(entries[VARIANTS[0]])
samples = np.zeros([n_samples, n_features])

i = 0
for e in VARIANTS:
    samples[i] = np.array(entries[e])
    i += 1

print(samples.shape)

dist_matrix = []

for i in range(n_samples):
    sample = samples[i]
    dist_matrix.append([])
    # for j in range(i + 1, n_samples):
    for j in range(n_samples):
        dist = scipy.spatial.distance.cityblock(sample, samples[j])
        # similarity = 1 - (dist / n_features)
        dist_matrix[i].append(dist / n_features)

dist_matrix = np.array(dist_matrix) * 100

fig, ax = plt.subplots()
im = ax.imshow(dist_matrix,
               cmap='magma_r',
               # Distinct cells instead of colours that bleed into one another:
               interpolation='nearest'
               )

# Annotate cells with percentage values.
for i in range(n_samples):
    for j in range(n_samples):
        ax.text(j, i, int(round(dist_matrix[i, j])),
                ha="center", va="center", color="w")


# Labels
ax.set_xticks(np.arange(n_samples))
ax.set_xticklabels(VARIANTS)
ax.set_yticks(np.arange(n_samples))
ax.set_yticklabels(VARIANTS)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.colorbar(im)

ax.set_title("Relative dialect distances (in %)")
fig.tight_layout()
plt.show()


linkage_matrix = scipy.cluster.hierarchy.linkage(samples,
                                                 method='average'  # UPGMA
                                                 # method='ward'
                                                 )

scipy.cluster.hierarchy.dendrogram(
    linkage_matrix,
    labels=VARIANTS,
    orientation='right',
    leaf_font_size=12.)
plt.show()
