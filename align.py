from lingpy.align.multiple import Multiple
from lingpy.sequence.sound_classes import token2class
from read_data import get_samples
from collections import Counter


def align_concept(doculects, doculects_cwg,
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
    print(len(labels), labels)
    assert len(sequences) == len(labels), ("The reference doculect needs to be"
                                           " in the dict of doculects")
    msa = Multiple(sequences, merge_geminates=True)
    if alignment_type == 'lib':
        msa.lib_align(mode=alignment_mode)
    else:
        msa.prog_align(mode=alignment_mode)
    alignments = msa.alm_matrix
    if verbose > 2:
        print(msa)

    if verbose > 1 and msa.swap_check():
        print("SWAPS!")
        print(alignments[0])
        print(msa.swap_index)

    if context_cv:
        ref = ['#'] + alignments[0] + ['#']
        ref_segments_cv = []
        for j in range(1, len(ref) - 1):
            r_prev = seg2class(ref[j - 1])
            r = ref[j]
            r_next = seg2class(ref[j + 1])
            ref_segments_cv.append((r_prev, r, r_next))
    if context_sc:
        ref = ['#'] + alignments[0] + ['#']
        ref_segments_sc = []
        for j in range(1, len(ref) - 1):
            r_prev = seg2class(ref[j - 1], sca=True)
            r = ref[j]
            r_next = seg2class(ref[j + 1], sca=True)
            ref_segments_sc.append((r_prev, r, r_next))

    corres = {}
    for i in range(1, len(labels)):
        if labels[i] not in doculects_cwg:
            continue
        corres_i = Counter()

        if no_context:
            c = zip(alignments[0], alignments[i])
            # Making the first part a tuple so corres.keys() can be sorted
            # later on if segments with contexts (-> always tuples) are also
            # included.
            corres_i.update([(tuple(x[0]), x[1]) for x in c
                             if x != ('-', '-')])

        ref_segments = []
        sca_model = []
        if context_cv:
            ref_segments.append(ref_segments_cv)
            sca_model.append(False)
        if context_sc:
            ref_segments.append(ref_segments_sc)
            sca_model.append(True)
        for ref_segs, sca in zip(ref_segments, sca_model):
            cur = ['#'] + alignments[i] + ['#']
            cur_segments = []
            for j in range(1, len(ref) - 1):
                c_prev = seg2class(cur[j - 1], sca=sca)
                c = cur[j]
                c_next = seg2class(cur[j + 1], sca=sca)
                cur_segments.append((c_prev, c, c_next))
            c = zip(ref_segs, cur_segments)
            corres_i.update([x for x in c if (x[0][1], x[1][1]) != ('-', '-')])

        corres[labels[i]] = corres_i
    return corres


def seg2class(segment, sca=False):
    if segment in ['#', '-']:
        return segment
    if sca:
        return token2class(segment, 'sca')
    cl = token2class(segment, 'dolgo')
    return 'V' if cl == 'V' else 'C'


def align(reference_doculect='ProtoGermanic', binary=True,
          alignment_type='lib', alignment_mode='global', min_count=0,
          no_context=True, context_cv=False, context_sc=False,
          verbose=1):
    if verbose > 0:
        print('Reading the data files.')
    entries, doculects_cwg, doculects_add = get_samples()
    doculects_cwg.remove(reference_doculect)
    correspondences = {}
    all_correspondences = Counter()

    if verbose > 0:
        print('Aligning the entries.')
    for concept, doculects in entries.items():
        if verbose > 2:
            print(concept)
        corres = align_concept(doculects, doculects_cwg,
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

    if min_count or verbose > 2:
        all_correspondences_old = all_correspondences.keys()
        all_correspondences = Counter()

        for doculect, tallies in correspondences.items():
            if binary:
                for corres in all_correspondences_old:
                    try:
                        if tallies[corres] < min_count:
                            del tallies[corres]
                    except KeyError:
                        pass
            all_correspondences.update(tallies)
            if verbose > 2:
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
    if verbose > 1:
        print(all_correspondences)
    print(len(doculects_cwg), doculects_cwg)
    return correspondences, all_correspondences, doculects_cwg
