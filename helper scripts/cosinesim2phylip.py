import pickle
import csv


def clean_node(text):
    return text.replace(' ', '_').replace('.', '') \
               .replace('(', '').replace(')', '') \
               .replace('ü', 'ue').replace('ë', 'e')


with open('output/cosinesim-context.pickle', 'rb') as f:
    sim = pickle.load(f)

doculects = []
with open('helper scripts/coordinates.csv', encoding='utf8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doculects.append(clean_node(row['doculect']))


with open('output/tfidf-context.dist', 'w', encoding='utf8') as f:
    f.write('\t' + str(len(sim)) + '\n')
    for doculect, sim_row in zip(doculects, sim):
        f.write(doculect)
        f.write('\t')
        f.write('\t'.join(['{:.4f}'.format(abs(1.0 - x)) for x in sim_row]))
        f.write('\n')
