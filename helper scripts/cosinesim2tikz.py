import pickle
import csv


def clean_node(text):
    return text.replace(' ', '').replace('.', '') \
               .replace('(', '').replace(')', '') \
               .replace('ü', 'ue').replace('ë', 'e')


def clean_label(text):
    return text.replace('ü', '\\"{u}').replace('ë', '\\"{e}')


al = {'Std. Dutch (NL)': 'Dutch Std NL', 'Std. Dutch (BE)': 'Dutch Std BE'}


def alias(text):
    try:
        return al[text]
    except KeyError:
        return text


with open('output/cosinesim-context.pickle', 'rb') as f:
    sim = pickle.load(f)

n = sim.shape[0]
min_sim, max_sim = -1, -1
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        d = sim[i, j]
        if min_sim == -1 or d < min_sim:
            min_sim = d
        if d > max_sim:
            max_sim = d

doculect2coord = {}
with open('helper scripts/coordinates.csv', encoding='utf8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doculect2coord[row['doculect']] = (float(row['longitude']) - 4,
                                           float(row['latitude']) - 46)

with open('doc/figures/cosine.tex', 'w', encoding='utf8') as f:
    f.write("\\documentclass{standalone}\n")
    f.write("\\usepackage{tikz}\n")
    f.write("\\usepackage[outline]{contour}\n")
    f.write("\\begin{document}\n")
    f.write("\\contourlength{0.1em}\n")
    f.write("\\begin{tikzpicture}[scale=2.5, dot/.style={draw, circle, "
            "minimum size=2mm, inner sep=0pt, outer sep=0pt, fill=red}]\n")
    for doc, loc in doculect2coord.items():
        f.write("\\node ({}) at {} {{}};\n"
                .format(clean_node(doc), loc))

    docs = {i: d for i, d in enumerate(sorted(doculect2coord.keys(),
                                              key=lambda x: alias(x)))}
    # Sort lines by similiarity score
    # -> print darker lines later.
    sims = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            # normalize similarity score
            d = (sim[i, j] - min_sim) / (max_sim - min_sim)
            line = "\\draw[line width={}mm, color=black!70!blue!{}]" \
                   "({}.center) -- ({}.center);\n" \
                   .format(d * 1.3, int(100 * d),
                           clean_node(docs[i]), clean_node(docs[j]))
            sims.append((d, line))
    for _, line in sorted(sims):
        f.write(line)

    for doc, loc in doculect2coord.items():
        f.write("\\node[dot] at ({}.center) {{}};\n"
                .format(clean_node(doc)))
        f.write("\\node[below] at ({}.center) "
                "{{\\contour{{white}}{{\\Large {}}}}};\n"
                .format(clean_node(doc), clean_label(doc)))
    f.write("\\end{tikzpicture}\n")
    f.write("\\end{document}\n")
