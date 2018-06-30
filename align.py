from lingpy.align.multiple import Multiple
from read_data import get_samples


entries = get_samples()
for concept, doculects in entries.items():
    print(concept)
    sequences = []
    for doculect, word in doculects.items():
        sequences.append(word)
    print(sequences)
    msa = Multiple(sequences)
    msa.prog_align()
    print(msa)
    print()
