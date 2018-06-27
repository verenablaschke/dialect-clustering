# Clustering dialect varieties based on historical sound correspondences

Can we cluster dialects based on sound correspondences? Papers like ```wieling2011bipartite``` invest clustering based on the presence/absence of aligned sound segments for doculect-reference doculect alignments, and give dialect clusters in addition to correlating the segment alignments with the clusters. Given the data I have, I will attempt something similar, but use Proto-Germanic data as reference doculect, effectively trying to get information about historical sound shifts.

Next steps: 
- Go carefully through publications about segment-correspondence-based clustering: ```wieling2011bipartite```, ```wieling2010hierarchical```, ```nerbonne2009data-driven```, ```dhillon2001co-clustering```, ```wieling2013analyzing```, ```prokic2010exploring```, ```montemagni2013synchronic```, ```wieling2014analyzing```. How exactly does this work, and why does it work? 
  - Are there authors other than Wieling and Nerbonne that have attempted something similar for language clustering? Are there publications more recent than 2014 about this?
  - Check conclusions about this (and other techniques) in [Advances in Dialectometry
Annual Review of Linguistics](https://www.annualreviews.org/doi/full/10.1146/annurev-linguist-030514-124930).
  - ```clustering_via_eigenvectors.py``` is an implementation of the example from ```wieling2011bipartite```
- Alignment: Proto-Germanic and (maybe) other doculects that were also transcribed by Maguire and shared on Heggarty's research project webpage(s). 


Notes:
- The method introduced in ```wieling2011bipartite``` is for flat clustering and a known number of clusters. ```wieling2010hierarchical``` is the hierarchical extension (the entire data set is the first cluster and then each cluster is recursively split into two clusters).
- ```wieling2011bipartite``` and ```wieling2010hierarchical``` consider affricates/diphthongs/triphthongs separate sound segments. Combining them into single multi-token segments might be more informative, especially since I expect the consequences of the High German consonant shift to be visible (incl. the *stop* > *affricate* shifts).
  - ```wieling2010hierarchical``` remark on a common alignment [-]:[ʃ], which commonly appears after [t]:[t]. Interpreting affricates as single segments with the result of correspondences such as [t]:[t͡ʃ], or using another approach to include contextual information seems more satisfying to me.

---

(The sections below haven't been updated yet.)

---

## Current Data

The BDPA contains a list of 111 cognate sets across 21 German/Dutch doculects, transcribed in IPA, tokenized (affricates and diphthongs constitute single segments), and already aligned. (More details in the ```data``` folder.) All of the entries were transcribed by a single person and aligned by another.

### TODO

- Unfortunately, the BDPA contains only two Low German dialects. Check if I can add some of the ones from [Sound Comparisons (Heggarty)](http://www.soundcomparisons.com/#/en/Germanic) (heggartynodatesound).
  - (Many of) the Germanic entries were also collected by Paul Heggarty and transcribed by Warren Maguire.
  - They aren't aligned with the other entries.
    - It would be possible however to start the program with an alignment step whose success is measured with the gold standard data.
    - If I add just one or two dialects, manual alignment might be worthwhile.
  - Some of the entries are marked "non-cognate", but this is also encoded in the corresponding CSV files.
- Possibly also check to what extend I might be able to use data from the Alignments of the Phonetischer Atlas von Deutschland (PAD) (prokic2018alignments), the Indo-European Lexical Cognacy Database (IELex) (dunn2012indo-european), the Morfologische Atlas van de Nederlandse Dialecten (MAND) (goemannodatemorfologische), or the Dynamische Syntactische Atlas van de Nederlandse Dialecten (DynaSAND) (barbiersnodatedynamische). Of course, for all of these, the transcription standards would probably differ from what I currently use.
- Maybe check if the non-IPA phonetic transcriptions of the Norwegian and Swedish parts of the [Nordic Dialect Corpus](http://www.tekstlab.uio.no/nota/scandiasyn/) are compatible with one another.
- The only difference between "High German (Herrlisheim)" and "High German (North Alsace)" is a slightly different coverage of entries. (That makes sense, since they seem to be derived from the same wordlist of Heggarty's.) Figure out if one is the subset of the other, else merge them?
  - [x] There are three differences between the two lists. "North Alsace" contains entries for _quick_ (408) and _top_ (430); "Herrlisheim" doesn't. The entries for _right_ (423) are "ʁaːχ" (North Alsace) and "ʁaːχt" (Herrlisheim). Heggarty's original entries on languagesandpeoples.com and soundcomparisons.com (both of which have only one doculect from North Alsace, which was recorded in Herrlisheim) are identical to the "Herrlisheim" entries. No idea where the data for "North Alsace" comes from, then (transcription errors?), so I'm disregarding that file now.
- I could also branch out by using all of the doculects with sufficient concept coverage from BDPA-Germanic. There are a bunch of varieties of English. Unfortunately not a lot of non-standard varieties for the other languages.

## Current implementation 

[cluster.py](https://github.com/verenablaschke/dialect-clustering/blob/master/cluster.py) currently contains a very rough implementation that does the following:
- For each doculect, concatenate all entries (including gap segments from the alignment) and convert the IPA segments into phonetic feature vectors using PanPhon, resulting in one very long feature vector per doculect.
- Get the Manhattan distance between each pair of doculects to create a distance matrix (visualized with the heatmap).
- Perform hierarchical clustering on the feature vectors using SciPiy's implementation of the UPGMA algorithm (Unweighted Pair Group Method using Arithmetic averages) (visualized with the dendrogram).

Interestingly, of the (currently only 2...) Low German dialects, one is grouped with the Dutch variants and the other is part of an otherwise High & Central German cluster.

Out of curiosity, I created a second dendrogram for all doculects (except for Proto-Germanic, for which I onyl have aligned data for 5 concepts). The overall clusters seem fine, although some of the subclusters within the Scandinavian cluster don't make sense to me.

### TODO

- Improve the feature vector conversion & distance measure:
  - [ ] Deal with diphthongs, triphthongs, affricates.
    - Used the PHOIBLE feature table instead of PanPhon's feature vectors in the branch ```phoible-features```. It recognizes some multi-phone segments, but it doesn't have entries for many of them either.
  - [ ] Escape ʦ, ʧ.
  - [x] If two doculects share a gap segment in an aligned entry, don't let that facture into the distance score.
  - [x] If an entry is missing for one of the words, should that entry be ignored for the distance measure? (In nerbonne1996phonetic missing entries are ignored, but I think I read other papers (which?) that didn't ignore such cases.)
  - Appropriate gap penalization.
  - The gap/missing word improvements currently only apply to the distance matrix, not to the clusters because of the input format SciPy expects. It might be worth looking into how complicated doing a re-implementation of UPGMA (or another hierarchical clustering algorithm) is.
- Try out different distance measures and clustering algorithms:
  - heeringa2006evaluation present and evaluate a bunch of different alignment/distance scoring strategies, such as including n-grams for phonetic context. See also nerbonne1997measuring.
  - heeringa2004measuring (ch. 6.1.3) compares different clustering algorithms (and ultimately prefers UPGMA)
  - multi-dimensional scaling (nerbonne2009data-driven, heeringa2004measuring ch. 6.2) and other dimensionality-reduction techniques prior to clustering
  - There are implementations where the focus is also on the features that distinguish doculect groups (prokic2012detecting, nerbonne2006identifying, wieling2011bipartite)
  - Bayesian phylogenetic inference? (ronquist2003mrbayes, prokicnodateinferring)
- average per-word distance, average per-doculect distance, standard error, significance level (see nerbonne1996phonetic)
- Figure out why LingPy crashes when trying to create a dendrogram from the data.
- Depending on how the clustering algorithms work, shuffling the order of the doculects might result in a slightly different cluster hierarchy.
- All of this is about the number of differences. It would also be interesting to consider the number of regular correspondences vs. unpredicatable correspondences/differences.
  - Is the size of the data sufficient for extracting regular sound correspondences?

## Evaluation

### TODO

- Literature research on 
  - (non-statistical) analyses of the German/Dutch(/Low German/Frisian) dialect landscape. Get some hierarchy that I can compare my tree to?
  - dialects vs. languages etc.
  - Low German as dialect vs language (perception)

## Notes

All of the comparisons here are phonetic (and might possibly include morphological information in some cases) and on a word level, but I'm ignoring lexical, syntactical, morphological, etc. information in this analysis.

# References

## Libraries

PanPhon v. 0.14 ([GitHub](https://github.com/dmort27/panphon), [PyPi](https://pypi.org/project/panphon/)):

- Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L. (2016). [**PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors.**](https://www.aclweb.org/anthology/C/C16/C16-1328.pdf) In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers* (pp. 3475-3484).

SciPy (cluster, distance)

## Data

List, J.-M. and Prokić, J. (2014). [**A benchmark database of phonetic alignments in historical linguistics and dialectology.**](https://pdfs.semanticscholar.org/4bd4/0ed75369e07756b338f81a9c9529e207e279.pdf) In: *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, 26—31 May 2014, Reykjavik. 288-294.

Renfrew, C. and Heggarty, P. (2009). **Languages and Origins in Europe.** URL: http://www.languagesandpeoples.com/.
