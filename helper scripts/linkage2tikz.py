# Transform a SciPy linkage matrix into a tikz (LaTeX) figure representing
# the clusters as a dendrogram.

import numpy as np
import pickle

DIST_MULTIPLIER_X = 15
DIST_MULTIPLIER_Y = 0.5


# Labels need to be in the order in which they will appear at the bottom of the
# figure.
# Cluster numbers are based on the label indices in the alphabetically sorted
# label array.
def transform(Z, labels, f):
    n_samples = len(labels)
    labels = [l.replace("_", " ") for l in labels]
    # Add IDs of new clusters as extra column.
    cluster_ids = np.arange(n_samples, n_samples + Z.shape[0]) \
                    .reshape(-1, 1)
    Z = np.hstack((Z, cluster_ids))

    label2cluster = {l: i for i, l in enumerate(sorted(labels))}
    cluster2xcoord = {label2cluster[l]: i + 1 for i, l in enumerate(labels)}

    # Leaf nodes.
    f.write("\\documentclass{standalone}\n")
    f.write("\\usepackage{tikz, tipa}\n")
    f.write("\\begin{document}\n")
    f.write("\\tikzset{corres/.style={fill=white, right=6mm, above=1mm,"
            "inner sep=0pt}}\n")
    # f.write("\\tikzset{doc/.append style={prefix after command={\\pgfextra"
    #         "{\\tikzset{every label/.style={rotate=-45}}}}}}\n") # TODO
    f.write("\\begin{tikzpicture}\n")
    f.write("% Doculects by singleton cluster number (alphabetical index).\n")
    for l in labels:
        i = label2cluster[l]
        # f.write("\\node[doc, label=right:{}] ({}) at ({}, 0) {{}};\n" # TODO
        f.write("\\node[label=left:{}] ({}) at (0, {}) {{}};\n"
                .format(l, i, cluster2xcoord[i] * DIST_MULTIPLIER_Y))

    # Branches.
    f.write("\n% Clusters from Z.\n")
    for row in Z:
        cl_1 = int(row[0])
        cl_2 = int(row[1])
        dist = row[2] * DIST_MULTIPLIER_X
        cl_new = int(row[4])
        y = (cluster2xcoord[cl_1] + cluster2xcoord[cl_2]) / 2
        cluster2xcoord[cl_new] = y
        f.write("\\node ({}) at ({},{}) {{}};\n"
                # .format(cl_new, x, dist)) # TODO
                .format(cl_new, dist, y * DIST_MULTIPLIER_Y))
        f.write("\\draw ({}) -| ({}.center);\n"
                .format(cl_1, cl_new))
        f.write("\\draw ({}) -| ({}.center);\n"
                .format(cl_2, cl_new))

    # Y axis.
    max_dist = round(dist) + 1
    f.write("\n% X axis.\n")
    f.write("\\draw[->] (0,0) -- node"
            # "[label={[rotate=90, label distance=1cm]90:Distance}]{{}} " +
            # "(0, {});\n".format(max_dist)) # TODO
            "[label={[label distance=-1.3cm]Cosine distance}]{{}} " +
            "({}, 0);\n".format(max_dist))
    # Not the most elegant way of performing this loop in LaTeX but it works.
    for x in np.arange(0, max_dist, 1.5):
        f.write("\\draw ({}, 1pt) -- ({}, -3pt) node"
                "[label={{[label distance=-7mm]{:.2f}}}] {{}};\n"
                .format(x, x, x / DIST_MULTIPLIER_X))

    f.write("\\end{tikzpicture}\n")
    f.write("\\end{document}\n")


with open('output/dendrogram-context.pickle', 'rb') as f:
    Z = pickle.load(f)
labels = [
    'Ortisei',
    'Herrlisheim',
    'Hard',

    'Walser',
    'Graubuenden',
    'Biel',

    'Tuebingen',
    'Cologne',
    'Luxembourg',

    'Westerkwartier',
    'Feer',
    'Heligoland',

    'Ostend',
    'Antwerp',
    'Dutch_Std_BE',

    'Veenkolonien',
    'Achterhoek',
    'Limburg',

    'Dutch_Std_NL',
    'Grou',
]
with open('doc/figures/tfidf-context.tex', 'w') as f:  # TODO
    transform(Z, labels, f)
