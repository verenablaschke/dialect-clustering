# Clustering dialect varieties based on historical sound correspondences

Can we cluster dialects based on sound correspondences? Papers like `wieling2011bipartite` invest clustering based on the presence/absence of aligned sound segments for doculect-reference doculect alignments, and give dialect clusters in addition to correlating the segment alignments with the clusters. Given the data I have, I will attempt something similar, but use Proto-Germanic data as reference doculect, effectively trying to get information about historical sound shifts.

## Data

The BDPA contains a list of 111 cognate sets across 21 German/Dutch doculects, transcribed in IPA, tokenized (affricates and diphthongs constitute single segments), and already aligned. (More details in the `data` folder.) All of the entries were transcribed by Warren Maguire (with revisions by Paul Heggarty), and aligned by another person (*who?*).
Heggarty's Sound Comparisons project contains further entries for the same cognates. The Germanic doculects I added from that project were also transcribed by Warren Maguire, but they are unaligned. 

Despite having been transcribed by the same person, there seem to be **noticeable transcription differences between the two data sets**, as evidenced by the fact that these tend to end up in singleton clusters. (Very striking with e.g. [`python align.py -t -c -d all](output/align-t-c-dall.log)`.)

- BDPA: The only difference between "High German (Herrlisheim)" and "High German (North Alsace)" is a slightly different coverage of entries. They seem to be derived from the same wordlist of Heggarty's.
- If I am re-aligning the entries anyway, why not use the Sound Comparisons versions of all the data? There are some slight inconsistensies between the two versions, despite having originally been transcribed by the same person (`ʦ` vs. `ts`, between two vowels: `ɪ` vs. `j`, inclusion of stress marks).
- [ ] Use either BDPA data or SoundComparisons data, but not both combined.

## The Project

### Alignments

At the moment, I do not use the gold-standard alignments from the BDPA because they only contain 4 (out of 111) Proto-Germanic entries. Instead, I use LingPy's SCA-based MSA method for (re-)aligning the data.

- [ ] Figure out a way of combining the alignments with Proto-Germanic with the gold-standard alignments. At least compare them to get an idea of how good the new alignments are?
- Re-aligning the BDPA yields much better results when including all Germanic doculects during the multi-sequence alignment step, compared to only using the DE/NL doculects that I use for clustering.
- I use LingPy's `lib_align(mode='global')` for the alignments. I tried out `lib_align` and `prog_align`, both with `mode='global'` and `mode='dialign'`, and based on quickly inspecting the output, `mode='dialign'` yields quite unsatisfying results for both alignment types while `mode='global'` seems fine for both of them. Including all available doculects in the alignment also appears to create better alignments than just using the DE-NL subset.
- [ ] Multi-token segments:
  - [x] LingPy treats diphthongs/triphthongs as single segments.
  - [x] LingPy treats geminates as single segments.
  - [ ] Affricates are only treated as single segments if they are connected with a tie bar or written as ligatures. The latter is the case for the BDPA data. The SoundComparisons data does not indicate affricates using either convention (see e.g.  `ts` in `soundcomparisons/westerkwartier.csv`), which results in them being treated as separate segments by LingPy. I now replace `ts` with `t͡s` although I have not yet double-checked if this is always appropriate.
    - [ ] Other transcription differences (between two vowels: `ɪ` vs. `j`).
- Option to exclude statistically insignificant/rare alignments.
  - Optional (console arg): Only including correspondences for a doculect if they occur at least `mincount` times in that doculect (as did `wieling2010hierarchical`). This makes sense intuively, but the threshold is of course somewhat arbitrary.
  - Optional (console arg): TF-IDF matrix instead of binary/count matrix.
  - Statistically insignificant alignments currently aren't excluded (most (all?) dialects exhibit correspondences like `n : n`, which don't seem very informative).
- `wieling2011bipartite` and `wieling2010hierarchical` consider affricates/diphthongs/triphthongs separate sound segments. Combining them into single multi-token segments might be more informative, especially since I expect the consequences of the High German consonant shift to be visible (incl. the *stop* > *affricate* shifts).
  - `wieling2010hierarchical` remark on a common alignment [-]:[ʃ], which commonly appears after [t]:[t]. Interpreting affricates as single segments with the result of correspondences such as [t]:[t͡ʃ], or using another approach to include contextual information seems more satisfying to me.
- Adding some phonetic context (e.g. `montemagni2013synchronic`). Is the segment preceded/followed by a consonant/vowel/word boundary? Using more refined sound classes could also be insightful.
  - What about a model where I use segments, segments with C/V/# contexts and segments with sound-class contexts? And then it's up to the model/the sound corresponding ranking metrics to figure out which information is relevant?

### Clustering

#### Bipartite spectral graph co-clustering

Relevant literature:

- introduced in [`dhillon2001co-clustering`](https://dl.acm.org/citation.cfm?doid=502512.502550)
- introduced as method for dialect clustering in [`wieling2011bipartite`](https://www.sciencedirect.com/science/article/pii/S0885230810000410?via%3Dihub)
- hierarchical extension: [`wieling2010hierarchical`](http://www.aclweb.org/anthology/W10-2305)
- application of the method: [`wieling2013analyzing`](https://academic.oup.com/dsh/article-lookup/doi/10.1093/llc/fqs047)
- application of the method + using left & right contexts for the sound segments: [`montemagni2013synchronic`](https://academic.oup.com/dsh/article-lookup/doi/10.1093/llc/fqs057)
- this article seems to introduce the same method as Dhillon's (or a very similar one) at the same time, in another journal: [`zha2001bipartite`](http://arxiv.org/abs/cs/0108018)
- this bioinformatics paper uses and describes the method as well: [`kluger2003spectral`](http://www.genome.org/cgi/doi/10.1101/gr.648603)

Steps:

1. Given a (binary) co-occurrence matrix `A (m x n; m = number of doculects, n = number of sound correspondences)`, **normalize this matrix**. First, create two diagonal matrices `D_1 (m x m)` and `D_2 (n x n)` that, respectively, contain the row sums/column sums of A. Compute the diagonal matrices' inverses. Since they are diagonal, this is very easy, but only possible if no entries on their diagonals are 0. Then, compute the square roots of the inverses (also easy because of the diagonal structure). Finally, create the normalized matrix `A_n = D_1 ^ -1/2 @ A @ D_2 ^ -1/2`. Effectively, you divide each entry by the square root of the sum of its row's entries and by the square root of the sum of its column's entries. This reduces the importance of doculects/correspondences that co-occur with a large number of other entries.
   - This goal doesn't seem too different from TF-IDF. After all, both are used for adjusting feature frequencies by how informative they are.

2. Perform **SVD** on `A_n` to get the left and right singular vectors `u_i` and `v_i`. We ignore the singular vectors belonging to the first/largest singular value, and take the second singular vectors (`u_2`, `v_2`). (If clustering with k > 2, also skip the first singular vectors and take the `log(k)/log2` following vectors.)
   - `kluger2003spectral` gives some more information as to why we're ignoring the first singular value/vectors (section "Independent Rescaling of Genes and Conditions"). Apparently, the first singular vectors only "make a trivial constant contribution to the matrix". I will need to carefully re-read that section to understand *why* this is. However, trying this out for the toy example in `wieling2011bipartite`, it is very much the case that when using the first singular vectors, the resulting vector (after step 3) contains the same value for all entries, being maximally unhelpful for k-means clustering (step 4).

3. Calculate `D_1 ^ -1/2 @ u_2` and `D_2 ^ -1/2 @ v_2`, and append them to get the vector/matrix Z. 

4. Perform **k-means** clustering on Z.

5. If performing hierarchical clustering, repeat steps 1-4 on all clusters individually.

#### Ranking sound correspondences by importance

Also introduced in [`wieling2011bipartite`](https://www.sciencedirect.com/science/article/pii/S0885230810000410?via%3Dihub).

For each cluster, rank the associated sound correspondences by the following metrics:

- **Representativeness**. How many doculects in this cluster exhibit this sound correspondence? `representativeness(cluster_i, corres_j) = number of doculects in cluster_i with corres_j / number of doculects in cluster_i `

- **Distinctiveness**. How often occurs a sound correspondence in this cluster compared to other clusters? This requires two additional measures: `relative_occurrence(cluster_i, corres_j) = number of doculects in cluster_i with corres_j / total number of doculects with corres_j` and `relative_size(cluster_i) = number of doculects in cluster_i / total number of doculects`. Then, `distinctiveness(cluster_i, corres_j) = (relative_occurrence(cluster_i, corres_j) - relative_size(cluster_i)) / (1 - relative_size(cluster_i))`.

- **Importance**. The average of representativeness and distinctiveness.

According to `wieling2010hierarchical`, the values of the **second right singular vector** are a good substitute for the above metrics.

#### Notes

- How beneficial is co-clustering really? Why not just cluster the doculects (after TF-IDF maybe) and then apply the sound correspondence metrics? It seems like the benefit of co-clustering is mainly that `v_2` can be used for ranking the correspondences, although this doesn't seem to be very popular among the authors who used bipartite spectral graph clustering for dialect data...

- I implemented the hierarchical version in the [`hierarchical` branch](https://github.com/verenablaschke/dialect-clustering/tree/hierarchical).  When trying to run it, I sometimes get a "singular matrix" error when trying to calculate the inverse of a matrix (for step 1 of the bipartite [...] clustering process).
  - This happens when the co-occurrence matrix for a cluster contains empty rows and/or columns. This shouldn't happen in the first case because empty rows/columns mean that the entries in question aren't similar to the other entries in the matrix, i.e. the previous clustering step performed quite poorly.
  - I'm working on a fix.

## Evaluation

I have the Glottolog codes and family tree information for the doculects (soundcomparisons.com provides Glottolog 2.7 codes for the samples). The only issue is that two of these codes aren't in Glottolog 3.2 (the current version) anymore. Why were they removed? Should I use the Glottolog 2.7 family tree information, or should I try to find appropriate substitute codes from Glottolog 3.2?

## Discussion

All of the comparisons here are phonetic (and might possibly include morphological information in some cases) and on a word level, but I'm ignoring lexical, syntactical, morphological, etc. information in this analysis.

Dialects -- hierarchy vs. web; 'vertical' changes vs. horizontal' influences.

- [ ] Read e.g. `heggarty2010splits`

## References

### Libraries

J.-M. List, S. Greenhill, and R. Forkel (2018):
**LingPy. A Python library for historical linguistics**. 
Version 2.6.3. 
URL: http://lingpy.org, 
DOI: https://zenodo.org/badge/latestdoi/5137/lingpy/lingpy. 
With contributions by Steven Moran, Peter Bouda, Johannes Dellert, Taraka Rama, Frank Nagel, and Tiago Tresoldi. 
Jena: Max Planck Institute for the Science of Human History.

### Data

J.-M. List and J. Prokić (2014). 
[**A benchmark database of phonetic alignments in historical linguistics and dialectology.**](https://pdfs.semanticscholar.org/4bd4/0ed75369e07756b338f81a9c9529e207e279.pdf)
In: *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, 26—31 May 2014, Reykjavik. 288-294.

C. Renfrew and P. Heggarty (2009).
**Languages and Origins in Europe.**
URL: http://www.languagesandpeoples.com/.

P. Heggarty 
**Sound Comparisons: Exploring Phonetic Diversity Across Language Families** 
URL: http://www.soundcomparisons.com/

H. Hammarström, S. Bank, R. Forkel, and M. Haspelmath (2018).
**Glottolog 3.2.**
Jena: Max Planck Institute for the Science of Human History.
URL: http://glottolog.org
