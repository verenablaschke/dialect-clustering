import os
import panphon
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
from matplotlib import pyplot as plt

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
# just for fun  # TODO remove?
DOCULECTS_ALL = ['American English', 'Australian English (Perth)',
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
FT = panphon.FeatureTable()
LEN_IPA_VEC = len(FT.fts('e').numeric())


def get_samples(directory, doculects=DOCULECTS):
    """"Extracts the feature-vector representations of each doculect's entries.

    Args:
        directory (str): The directory containing MSA files from the BDPA.

    Keyword args:
        doculects (list(str)): The names of the doculects for which information
                               should be extracted. (default: DOCULECTS)

    Returns:
        list(list(list(int))): A 4D matrix. The first (=outermost) dimension
                               represents doculects, the next one concepts,
                               then phonetic segments, then phonetic features.
                               The second and third dimensions can also contain
                               None values.
        list(int): The number of segments (including gaps) each concept
                   consists of.
    """
    entries = {}
    word_lengths = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.msa'):
                subentries, word_length = parse_file(os.path.join(root, f),
                                                     doculects)
                if subentries is None:
                    continue
                for e in subentries:
                    try:
                        entries[e].append(subentries[e])
                    except KeyError:
                        entries[e] = [subentries[e]]
                word_lengths.append(word_length)
    return [entries[v] for v in doculects], word_lengths


def parse_file(filename, doculects):
    """Extracts the phonetic segments as features vectors from a file.

    Args:
        filename (str): The MSA file.
        doculects (list(str)): The names of the relevant doculects.

    Returns:
        dict(str -> list(list(int))): A dictionary from doculect names to a 3D
                                      matrix. The first (=outermost) dimension
                                      represents concepts, the next one
                                      phonetic segments, and the last one
                                      phonetic features. The lists can also be
                                      None.
        list(int): The number of segments (including gaps) each concept
                   consists of.
    """
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
        # Other lines: language_name......\tsegment_1\tsegment_2\tsegment_n
        entries = {}
        for line in lines[2:]:
            cells = line.split('\t')
            doculect = cells[0].replace('.', '')
            if doculect in doculects:
                entries[doculect] = tokens2vec(cells[1:])
            elif doculect == 'LOCAL':
                # 'LOCAL' is present in all MSA files to summarize the segment/
                # gap distribution across the entries for the concept.
                word_length = len(cells[1:])
        for doculect in doculects:
            try:
                entries[doculect]
            except KeyError:
                print('{} has no entry for {}.'.format(doculect, concept))
                entries[doculect] = None
    return entries, word_length


def tokens2vec(segments):
    """Converts each phonetic segment in a list into a phonetic feature vector.

    Args:
        segments (list(str)): A list of phonetic segments, in IPA.

    Returns:
        list(list(int)): A 2D matrix. The first dimension describes concepts,
                         the second one segments as numeric phonetic features.
                         The first list may contain None entries if a segment
                         has no phonetic feature representation.
    """
    segs = []
    for seg in segments:
        # TODO: diphthongs & triphthongs! affricates!
        try:
            segs.append(FT.fts(seg).numeric())
        except AttributeError:
            # Couldn't convert the IPA token,
            # probably because it's an insertion/deletion dummy token ('-').
            if seg != '-':  # and len(seg) == 1:
                # print("Couldn't convert '{}' in {}."
                #       .format(seg, ''.join(segments)))
                pass  # TODO del
            segs.append(None)
    return segs


def distance(doculect1, doculect2,
             gap_penalty=LEN_IPA_VEC, ignore_missing_entries=True):
    """Calculates the relative distance between two doculects.

    Args:
        doculect1 (str): The name of the first doculect.
        doculect2 (str): The name of the second doculect.

    Keyword args:
        gap_penalty (int/float): The cost of insertion/deletion.
                                 (default: LEN_IPA_VEC)
        ignore_missing_entries (bool): If true, ignores cases where one of the
                                       doculect does not have an entry for a
                                       concept.

    Returns:
        float: The distance score [0, 1].
    """
    assert len(doculect1) == len(doculect2)
    len_vecs = 0
    dist = 0
    for vec1, vec2 in zip(doculect1, doculect2):
        if vec1 is None and vec2 is None:
            # Neither doculect has an entry for the concept.
            continue
        if vec1 is None or vec2 is None:
            if ignore_missing_entries:
                continue
            # TODO

        for seg1, seg2 in zip(vec1, vec2):
            len_vecs += 1
            if seg1 is None and seg2 is None:
                # Ignore gap-gap alignments.
                continue
            if seg1 is None or seg2 is None:
                # TODO: how heavily should gaps be penalized?
                dist += gap_penalty
                continue
            # TODO try out other distance measures?
            dist += scipy.spatial.distance.cityblock(seg1, seg2)

    len_vecs *= LEN_IPA_VEC
    # TODO check if the score can reach 1
    return dist / len_vecs


def numeric_null_segment():
    """Creates a dummy feature vector.

    Returns:
        list(str): A phonetic feature vector containing only 0s.
    """
    # TODO better approach?
    return [0 for _ in range(LEN_IPA_VEC)]


def distance_matrix(samples, visualize=True, doculects=DOCULECTS):
    """Creates a distance matrix between all doculects.

    Args:
        samples (list(list(list(int)))): A 4D matrix (doculects, concepts,
                                         segments, features). See get_samples.

    Keyword args:
        visualize (bool): If true, the distance matrix is displayed as a
                          heatmap. (default: True)
        doculects (list(str)): The names of the doculects (only needed if
                               visualize is True). (default: DOCULECTS)

    Returns:
        np.array: A n_doculects x n_doculects matrix containing the doculect
                  distances.
    """
    dist_matrix = []
    n_samples = len(samples)
    for i in range(n_samples):
        sample = samples[i]
        dist_matrix.append([])
        for j in range(n_samples):
            if i == j:
                # Distance between a doculect and itself.
                dist = 0
            elif j < i + 1:
                # Our distance measure is commutative; we've already calculated
                # this distance.
                dist = dist_matrix[j][i]
            else:
                dist = distance(sample, samples[j])
            dist_matrix[i].append(dist)
    dist_matrix = np.array(dist_matrix)

    if visualize:
        # Heatmap.
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
        ax.set_xticklabels(doculects)
        ax.set_yticks(np.arange(n_samples))
        ax.set_yticklabels(doculects)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.colorbar(im)

        ax.set_title("Relative doculect distances (in %)")
        fig.tight_layout()
        plt.show()

    return dist_matrix


def flatten_matrix(samples, word_lengths):
    """Flattens the matrix to 2D and replaces None segments.

    Args:
        samples (list(list(list(int)))): A 4D matrix (doculects, concepts,
                                         segments, features). See get_samples.
        word_lengths (list(int)): The number of segments (including gaps) each
                                  concept consists of.

    Returns:
        np.array: An n_doculects x n_features matrix.
    """
    samples_flat = []
    for doculect in samples:
        vec_doculect = []
        i = 0
        for word in doculect:
            i += 1
            vec_word = []
            if word is None:
                # TODO check if this changes the actual matrix.
                word = [numeric_null_segment()
                        for _ in range(word_lengths[i - 1])]
            for vec in word:
                if vec is None:
                    vec_word += numeric_null_segment()
                else:
                    vec_word += vec
            vec_doculect += vec_word
        samples_flat.append(vec_doculect)

    # Convert to NumPy array.
    samples_np = np.zeros([len(samples_flat), len(samples_flat[0])])
    i = 0
    for entry in samples_flat:
        samples_np[i] = np.array(entry)
        i += 1

    return samples_np


def cluster(samples, method='average', visualize=True):
    """Performs hierarchical clustering on the doculects.

    Args:
        samples (np.array): An n_doculects x n_features matrix.

    Keyword args:
        method (str): The clustering method
                      (see scipy.cluster.hierarchy.linkage).
        visualize (bool): If true, displays the cluster hierarchy as a
                          dendrogram.

    Returns:
        np.array: The linkage matrix encoding the hierarchical clusters.
    """
    # 'average' corresponds to the UPGMA clustering algorithm
    linkage_matrix = scipy.cluster.hierarchy.linkage(samples, method=method)
    if visualize:
        scipy.cluster.hierarchy.dendrogram(
            linkage_matrix,
            labels=DOCULECTS,
            orientation='right',
            leaf_font_size=12.)
        # TODO label x axis
        plt.show()
    return linkage_matrix


if __name__ == '__main__':

    samples, word_lengths = get_samples('data/bdpa')
    distance_matrix(samples)
    samples = flatten_matrix(samples, word_lengths)
    cluster(samples)
