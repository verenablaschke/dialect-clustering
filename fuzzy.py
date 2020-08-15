import numpy as np
np.set_printoptions(precision=4, suppress=True)


def find_fuzzy_c_means(data, k_min, k_max, doculects, m=1.5,
                       filename=None, max_iter=1000):
    for k_i in range(k_min, k_max + 1):
        partitions, centroids = fuzzy_c_means(data, k_i, doculects, m,
                                              filename, max_iter)
        # Get each doculect's distance to its highest-scoring cluster.
        dist = 0
        for datum, part in zip(data, partitions):
            centre = centroids[np.argmax(part)]
            dist += np.sum([(d - c) ** 2 for d, c in
                           zip(datum, centre)]) ** 0.5
        print(k_i, dist)


def fuzzy_c_means(data, k, doculects, m=1.5, filename=None, max_iter=1000):
    # m = 2 tends to produce a partition matrix where every entry is 1 / k

    n = data.shape[0]
    f = data.shape[1]
    partitions = np.random.rand(n, k)
    partitions /= np.sum(partitions, axis=1, keepdims=1)

    min_change = 0.0001
    change = 1
    for i in range(max_iter):
        centroids = update_centroids(data, partitions, n, f, k)
        new_partitions = update_partitions(data, centroids, n, k, f, m)
        change = np.sum(abs(partitions - new_partitions))
        if change < min_change:
            break
        partitions = new_partitions
    print("Found a partitioning after {} iterations "
          "(change to previous: {:.4f})".format(i, change))
    if filename:
        print_to_log(partitions, doculects, k, m, i, filename)
    return partitions, centroids


def update_partitions(data, centroids, n, k, f, m):
    if m == 1:
        m = 1.01
    exponent = 2 / (m - 1)  # TODO convert to log?
    partitions = np.zeros((n, k))
    for i in range(n):
        dists = [np.sum([(da - ce) ** 2 for da, ce in
                         zip(data[i], centroids[c])]) ** 0.5
                 for c in range(k)]
        for j in range(k):
            if dists[j] == 0:
                partitions[i][j] = 0
                continue
            div = 0
            for j2 in range(k):
                div += (dists[j] / dists[j2]) ** exponent
            partitions[i][j] = 1 / div
    return partitions


def update_centroids(data, partitions, n, f, k):
    centroids = []
    for c in range(k):
        centroid = np.zeros((n, f))
        div = 0
        for i in range(n):
            membership = partitions[i][c] ** 2
            centroid[i] = membership * data[i]
            div += membership
        centroid = np.sum(centroid, axis=0) / div
        centroids.append(centroid)
    return centroids


def print_to_log(partitions, doculects, k, m, n_iter, filename):
    entries = sorted([(np.argmax(part), doc, part)
                      for doc, part in zip(doculects, partitions)],
                     key=lambda x: (x[0], -x[2][x[0]]))
    with open(filename, 'w', encoding='utf8') as f:
        f.write("k={}, m={}\n".format(k, m))
        f.write("Found a partitioning after {} iterations.\n\n".format(n_iter))
        prev_argmax = entries[0][0]
        for argmax, doc, part in entries:
            if prev_argmax != argmax:
                f.write('\n')
            prev_argmax = argmax
            f.write(doc)
            f.write('\t')
            f.write(str(part))
            f.write('\n')


def print_to_tex(partitions, doculects, k, filename):
    entries = sorted([(np.argmax(part), doc, part)
                      for doc, part in zip(doculects, partitions)],
                     key=lambda x: (x[0], -x[2][x[0]]))
    doculects_tex = {'Dutch_Std_BE': 'Std. Dutch (BE)',
                     'Dutch_Std_NL': 'Std. Dutch (NL)',
                     'Graubuenden': 'Graub\\"{u}nden',
                     'Tuebingen': 'T\\"{u}bingen',
                     'Veenkolonien': 'Veenkoloni\\"{e}n'}
    with open(filename, 'w', encoding='utf8') as f:
        f.write("% Requires the packages booktabs, xcolor, calc.\n")
        f.write("% Based on: https://tex.stackexchange.com/a/445291\n")
        f.write("\\newlength\\BARWIDTH\n\\setlength\\BARWIDTH{1cm}\n")
        f.write("\\def\\bwbar#1{%%\n#1 {\\color{black!100}"
                "\\rule{#1cm}{8pt}}{\\color{black!30}\\rule{"
                "\\BARWIDTH - #1 cm}{8pt}}}\n")
        f.write("\\begin{tabular}{l " + ' '.join(['r' for _ in range(k)]))
        f.write("}\n\\toprule\n")
        f.write("Doculect & ")
        f.write(' & '.join(['Cluster ' + chr(ord('@') + (c + 1))
                            for c in range(k)]))
        f.write("\\\\\\midrule\n")
        for argmax, doc, part in entries:
            f.write(doculects_tex.get(doc, doc) + ' & ')
            f.write(' & '.join(["\\bwbar{" + "{:.2f}".format(p) + "}"
                                for p in part]))
            f.write('\\\\\n')
        f.write("\\bottomrule\n\\end{tabular}\n")
