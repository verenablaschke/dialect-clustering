import os
import panphon
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
from matplotlib import pyplot as plt

DATA_DIR = 'data/bdpa'
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
            # including Yiddish just out of curiosity
            'Yiddish (New York)'
            ]
# just for fun  # TODO remove?
VARIANTS_ALL = ['American English', 'Australian English (Perth)',
                'Belgian Dutch', 'Canadian English',
                'Central German (Cologne)', 'Central German (Honigberg)',
                'Central German (Luxembourg)', 'Central German (Murrhardt)',
                'Danish', 'Dutch', 'Dutch (Antwerp)', 'Dutch (Limburg)',
                'Dutch (Ostend)', 'English', 'English (Buckie)',
                'English (Lindisfarne)', 'English (Liverpool',
                'English (London', 'English (North Carolina)',
                'English (Singapore)', 'English (Tyrone)', 'Faroese',
                'German', 'High German (Biel)', 'High German (Bodensee)',
                'High German (Graubuenden)', 'High German (Herrlisheim)',
                'High German (North Alsace)', 'High German (Ortisei)',
                'High German (Tuebingen)', 'High German (Walser)',
                'Icelandic', 'Indian English (Delhi)',
                'Low German (Achterhoek)', 'Low German (Bargstedt)',
                'New Zealand English (Auckland)', 'Nigerian English (Igbo)',
                'Norwegian (Stavanger)', 'Scottish',
                'South African English (Johannisburg)', 'Swedish (Skane)',
                'Swedish (Stockholm)', 'West Frisian (Grou)',
                'Yiddish (New York)']
# VARIANTS = VARIANTS_ALL  # TODO remove
ft = panphon.FeatureTable()
LEN_IPA_VEC = len(ft.fts('e').numeric())


def parse_file(filename):
    with open(filename, encoding='utf8') as f:
        lines = [line.replace('.', '').replace('\n', '')
                 for line in f.readlines()]
        # Line 0: language family
        # Line 1: term (in EN)
        concept = lines[1].replace('"', '')
        # if concept == 'tear':
        #     # 'tear' is divided into two files, each with a different concept
        #     # set that only covers about half the doculects.
        #     return None
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
            segs.append(ft.fts(token).numeric())
        except AttributeError:
            # Couldn't convert the IPA token,
            # probably because it's an insertion/deletion dummy token.
            segs.append(None)
    return segs


def distance(dialect1, dialect2):
    assert len(dialect1) == len(dialect2)
    len_vecs = 0
    dist = 0
    for vec1, vec2 in zip(dialect1, dialect2):
        if vec1 is None and vec2 is None:
            continue
        len_vecs += 1
        if vec1 is None or vec2 is None:
            # TODO: should gaps be penalized this heavily?
            dist += LEN_IPA_VEC
            continue
        dist += scipy.spatial.distance.cityblock(vec1, vec2)
    len_vecs *= LEN_IPA_VEC
    return dist / len_vecs


def numeric_null_segment():
    # TODO better approach?
    return [0 for _ in range(LEN_IPA_VEC)]


def distance_matrix(samples, n_samples, visualize=True):
    dist_matrix = []
    for i in range(n_samples):
        sample = samples[i]
        dist_matrix.append([])
        for j in range(n_samples):
            if i == j:
                dist = 0
            elif j < i + 1:
                dist = dist_matrix[j][i]
            else:
                dist = distance(sample, samples[j])
            dist_matrix[i].append(dist)

    dist_matrix = np.array(dist_matrix)

    if visualize:
        # Heat map.
        fig, ax = plt.subplots()
        dist_matrix_percentage = dist_matrix * 100
        im = ax.imshow(dist_matrix_percentage,
                       cmap='magma_r',
                       interpolation='nearest'  # distinct cells
                       )

        # Annotate cells with percentage values.
        for i in range(n_samples):
            for j in range(n_samples):
                ax.text(j, i, int(round(dist_matrix_percentage[i, j])),
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

    return dist_matrix


def flatten_matrix(samples):
    # Flatten the matrix to 2D and replace None segments.
    samples_flat = []
    for dialect in samples:
        vec_long = []
        for vec in dialect:
            if vec is None:
                vec_long += numeric_null_segment()
            else:
                vec_long += vec
        samples_flat.append(vec_long)

    # Convert to NumPy array.
    n_features = len(samples_flat[0])
    samples_np = np.zeros([n_samples, n_features])
    i = 0
    for entry in samples_flat:
        samples_np[i] = np.array(entry)
        i += 1

    return samples_np


def cluster(samples, method='average', visualize=True):
    # 'average' corresponds to the UPGMA clustering algorithm
    linkage_matrix = scipy.cluster.hierarchy.linkage(samples, method=method)
    if visualize:
        scipy.cluster.hierarchy.dendrogram(
            linkage_matrix,
            labels=VARIANTS,
            orientation='right',
            leaf_font_size=12.)
        plt.show()
        # TODO label x axis
    return linkage_matrix


# Represent the dialects as feature vectors.
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


# Create a distance matrix.
n_samples = len(VARIANTS)
samples = [entries[v] for v in VARIANTS]
distance_matrix(samples, n_samples)
samples = flatten_matrix(samples)
cluster(samples)
