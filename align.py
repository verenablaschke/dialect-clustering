from lingpy.align.multiple import Multiple
from lingpy.sequence.sound_classes import token2class
from read_data import get_samples
from collections import Counter


def align_concept(doculects, doculects_cwg, f, corres2lang2word=None,
                  reference_doculect='ProtoGermanic',
                  alignment_type='lib', alignment_mode='global',
                  no_context=True, context_cv=False, context_sc=False,
                  verbose=1):
    sequences = []
    labels = [reference_doculect]
    for doculect, word in doculects.items():
        if doculect == reference_doculect:
            sequences.insert(0, word)
        else:
            sequences.append(word)
            labels.append(doculect)
    assert len(sequences) == len(labels), ("The reference doculect needs to be"
                                           " in the dict of doculects")
    msa = Multiple(sequences, merge_geminates=True)
    if alignment_type == 'lib':
        msa.lib_align(mode=alignment_mode)
    else:
        msa.prog_align(mode=alignment_mode)
    alignments = msa.alm_matrix
    if verbose > 3:
        for line, label in zip(alignments, labels):
            print("{:15s} {}".format(label, '\t'.join(line)))
        print()

    if verbose > 1 and msa.swap_check():
        print("SWAPS!")
        print(msa.swap_index)
        print()

    # Remove doculects that were only used for alignment.
    msa_cwg = []
    labels_cwg = []
    for line, label in zip(alignments, labels):
        if label in doculects_cwg or label == reference_doculect:
            labels_cwg.append(label)
            msa_cwg.append(line)
            f.write("{:15s} {}\n".format(label, '\t'.join(line)))
    f.write("\n")

    # Padding in case context_cv or context_sc is True.
    ref = ['#'] + msa_cwg[0] + ['#']
    corres = {}
    if corres2lang2word is None:
        corres2lang2word = {}
    for i in range(1, len(labels_cwg)):
        corres_i = Counter()
        cur = ['#'] + msa_cwg[i] + ['#']

        # Extract the sound correspondences from the sequences.
        for c in range(1, len(ref) - 1):
            ref_i = ref[c]
            cur_i = cur[c]
            if ref_i == cur_i == '-':
                # Ignore gap-gap alignments.
                continue
            if no_context:
                corres_i.update([(ref_i, cur_i)])
            if context_cv or context_sc:
                left = ref[c - 1]
                if left != cur[c - 1]:
                    # Use context information only if it can be made to form
                    # a phonological rule (i.e. the context is the same in both
                    # the reference and modern doculect).
                    left = None
                offset = 2
                while left == '-':
                    # If the context is a gap for both doculects, get the
                    # nearest left context that is not a gap for at least one
                    # of the doculects.
                    left = ref[c - offset]
                    if left != cur[c - offset]:
                        left = None
                    offset += 1
                right = ref[c + 1]
                if right != cur[c + 1]:
                    right = None
                offset = 2
                while right == '-':
                    # If the context is a gap for both doculects, get the
                    # nearest right context that is not a gap for at least one
                    # of the doculects.
                    right = ref[c + offset]
                    if right != cur[c + offset]:
                        right = None
                    offset += 1
                if context_cv and left:
                    corres_i.update([(ref_i, cur_i, "{} _".format(
                        seg2class(left)))])
                if context_cv and right:
                    corres_i.update([(ref_i, cur_i, "_ {}".format(
                        seg2class(right)))])
                if (context_sc and left and
                        not (context_cv and left == '#')):
                    # Don't add sound class-independent context
                    # information (-> word boundaries) twice.
                    corres_i.update([(ref_i, cur_i, "{} _".format(
                        seg2class(left, sca=True)))])
                if (context_sc and right and
                        not (context_cv and right == '#')):
                    corres_i.update([(ref_i, cur_i, "_ {}".format(
                        seg2class(right, sca=True)))])
        # End character-level correspondence extraction.

        d = labels_cwg[i]
        corres[d] = corres_i
        for c in corres_i:
            try:
                # Use doculects[d] instead of sequences[i] because LingPy
                # changes the contents of 'sequences'.
                corres2lang2word[c][d].append(doculects[d])
            except KeyError:
                try:
                    corres2lang2word[c][d] = [doculects[d]]
                except KeyError:
                    corres2lang2word[c] = {d: [doculects[d]]}
    return corres, corres2lang2word


def seg2class(segment, sca=False):
    if segment in ['#', '-']:
        return segment
    if sca:
        return token2class(segment, 'sca')
    cl = token2class(segment, 'dolgo')
    return 'vowel' if cl == 'V' else 'cons'


def align(reference_doculect='ProtoGermanic',
          alignment_type='lib', alignment_mode='global', min_count=0,
          no_context=True, context_cv=False, context_sc=False,
          verbose=1):
    f = open('output/alignments.txt', 'w', encoding='utf8')
    if verbose > 0:
        print('Reading the data files.')
    entries, doculects_cwg, doculects_add = get_samples()
    doculects_cwg.remove(reference_doculect)
    correspondences = {}
    all_correspondences = Counter()

    if verbose > 0:
        print('Aligning the entries.')
    corres2lang2word = None
    for concept, doculects in entries.items():
        f.write("{}\n".format(concept))
        corres, corres2lang2word = align_concept(doculects, doculects_cwg, f,
            corres2lang2word=corres2lang2word,
            reference_doculect=reference_doculect,
            alignment_type=alignment_type,
            alignment_mode=alignment_mode,
            no_context=no_context, context_cv=context_cv,
            context_sc=context_sc,
            verbose=verbose)
        for doculect, tallies in corres.items():
            all_correspondences.update(tallies)
            try:
                correspondences[doculect].update(tallies)
            except KeyError:
                correspondences[doculect] = tallies
    f.close()

    with open('output/corres.txt', 'w', encoding='utf8') as f:
        all_correspondences_old = all_correspondences.keys()
        all_correspondences = Counter()

        for doculect, tallies in correspondences.items():
            f.write("Doculect: {}\n\n".format(doculect))
            f.write(str(tallies.items()))
            f.write("\n=================================================\n\n")
            for corres in all_correspondences_old:
                try:
                    # Remove rare correspondences.
                    if tallies[corres] < min_count:
                        del tallies[corres]
                except KeyError:
                    pass
            all_correspondences.update(tallies)
            if verbose > 3:
                print(doculect)
                print(tallies)
                print()

    all_correspondences = sorted(all_correspondences.keys())
    try:
        # In case there are any null-to-null alignments because of a larger
        # set of doculects being used for the alignment than for the tallies.
        all_correspondences.remove(('-', '-'))
    except ValueError:
        pass
    if verbose > 2:
        print(corres2lang2word)
    return (correspondences, all_correspondences,
            doculects_cwg, corres2lang2word)


if __name__ == "__main__":
    align(verbose=3)
