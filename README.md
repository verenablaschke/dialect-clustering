# Clustering dialect varieties based on historical sound correspondences

Can we cluster dialects based on sound correspondences? Papers like ```wieling2011bipartite``` invest clustering based on the presence/absence of aligned sound segments for doculect-reference doculect alignments, and give dialect clusters in addition to correlating the segment alignments with the clusters. Given the data I have, I will attempt something similar, but use Proto-Germanic data as reference doculect, effectively trying to get information about historical sound shifts.

## Data

The BDPA contains a list of 111 cognate sets across 21 German/Dutch doculects, transcribed in IPA, tokenized (affricates and diphthongs constitute single segments), and already aligned. (More details in the ```data``` folder.) All of the entries were transcribed by Warren Maguire (with revisions by Paul Heggarty), and aligned by another person (*who?*).
Heggarty's Sound Comparisons project contains further entries for the same cognates. The Germanic doculects I added from that project were also transcribed by Warren Maguire, but they are unaligned. 

- The only difference between "High German (Herrlisheim)" and "High German (North Alsace)" is a slightly different coverage of entries. (That makes sense, since they seem to be derived from the same wordlist of Heggarty's.) Figure out if one is the subset of the other, else merge them?
  - [x] There are three differences between the two lists. "North Alsace" contains entries for _quick_ (408) and _top_ (430); "Herrlisheim" doesn't. The entries for _right_ (423) are "ʁaːχ" (North Alsace) and "ʁaːχt" (Herrlisheim). Heggarty's original entries on languagesandpeoples.com and soundcomparisons.com (both of which have only one doculect from North Alsace, which was recorded in Herrlisheim) are identical to the "Herrlisheim" entries. No idea where the data for "North Alsace" comes from, then (transcription errors?), so I'm disregarding that file now.
- I could also branch out by using all of the doculects with sufficient concept coverage from BDPA-Germanic. There are a bunch of varieties of English. Unfortunately not a lot of non-standard varieties for the other languages.

## The Project

### Alignments

At the moment, I do not use the gold-standard alignments from the BDPA because they only contain 4 (out of 111) Proto-Germanic entries. Instead, I use LingPy's SCA-based MSA method for (re-)aligning the data.

- [ ] Figure out a way of combining the alignments with Proto-Germanic with the gold-standard alignments. At least compare them to get an idea of how good the new alignments are?
- Based on a brief look at the alignment tables created in ```align.py```, it appears that the P-G suffixes cause mis-alignments where root segments from the other doculects are sometimes aligned with the P-G suffix segments rather than the root segments.
- [ ] Multi-token segments:
  - [x] LingPy treats diphthongs/triphthongs as single segments.
  - [x] LingPy treats geminates as single segments.
  - [ ] Affricates are only treated as single segments if they are connected with a tie bar or written as ligatures. The latter is the case for the BDPA data. The SoundComparisons data does not indicate affricates using either convention (see e.g.  `ts` in `soundcomparisons/westerkwartier.csv`), which results in them being treated as separate segments by LingPy.
- [ ] Exclude statistically insignificant/rare alignments.
  - Currently only including correspondences for a doculect if they occur at least three times in that doculect (as did `wieling2010hierarchical`). This makes sense intuively, but the threshold is of course somewhat arbitrary.
  - Statistically insignificant alignments currently aren't excluded (most (all?) dialects exhibit correspondences like `n : n`, which don't seem very informative).

Considering diphthongs/triphthongs/affricates/geminates single segments should yield more informative correspondences. Can we add more phonetic context though? 

### Clustering

- Go carefully through publications about segment-correspondence-based clustering: ```wieling2011bipartite```, ```wieling2010hierarchical```, ```nerbonne2009data-driven```, ```dhillon2001co-clustering```, ```wieling2013analyzing```, ```prokic2010exploring```, ```montemagni2013synchronic```, ```wieling2014analyzing```. How exactly does this work, and why does it work? 
  - Are there authors other than Wieling and Nerbonne that have attempted something similar for language clustering? Are there publications more recent than 2014 about this?
  - Check conclusions about this (and other techniques) in [Advances in Dialectometry
Annual Review of Linguistics](https://www.annualreviews.org/doi/full/10.1146/annurev-linguist-030514-124930).
  - ```clustering_via_eigenvectors.py``` is an implementation of the example from ```wieling2011bipartite```.
  - The clustering of the BDPA/SoundComparisons data is currently performed in `align.py`.
  - The co-clustering of doculects and features currently doesn't seem to work that great. I (arbitrarily) picked k=5 clusters. One of these clusters is only picked for a couple of sound correspondences but for no dialects (see `align.log`). Why? Is it the number of clusters? What are the particularities of the correspondences associated with the otherwise empty clusters?


Notes:
- The method introduced in ```wieling2011bipartite``` is for flat clustering and a known number of clusters. ```wieling2010hierarchical``` is the hierarchical extension (the entire data set is the first cluster and then each cluster is recursively split into two clusters).
- ```wieling2011bipartite``` and ```wieling2010hierarchical``` consider affricates/diphthongs/triphthongs separate sound segments. Combining them into single multi-token segments might be more informative, especially since I expect the consequences of the High German consonant shift to be visible (incl. the *stop* > *affricate* shifts).
  - ```wieling2010hierarchical``` remark on a common alignment [-]:[ʃ], which commonly appears after [t]:[t]. Interpreting affricates as single segments with the result of correspondences such as [t]:[t͡ʃ], or using another approach to include contextual information seems more satisfying to me. (see previous section)

Additionally?
- Analysis of eigenvectors/PMI-based analysis to rank correspondences by importance?

## Evaluation

- Literature research on (non-statistical) analyses of the German/Dutch(/Low German/Frisian) dialect landscape. Get some hierarchy that I can compare my tree to?

## Notes

All of the comparisons here are phonetic (and might possibly include morphological information in some cases) and on a word level, but I'm ignoring lexical, syntactical, morphological, etc. information in this analysis.

Dialects -- hierarchy vs. web; 'vertical' changes vs. horizontal' influences.

- [ ] Read e.g. `heggarty2010splits`

## References

### Libraries

List, Johann-Mattis; Greenhill, Simon; and Forkel, Robert (2018): **LingPy. A Python library for historical linguistics**. Version 2.6.3. URL: http://lingpy.org, DOI: https://zenodo.org/badge/latestdoi/5137/lingpy/lingpy. With contributions by Steven Moran, Peter Bouda, Johannes Dellert, Taraka Rama, Frank Nagel, and Tiago Tresoldi. Jena: Max Planck Institute for the Science of Human History.

### Data

List, J.-M. and Prokić, J. (2014). [**A benchmark database of phonetic alignments in historical linguistics and dialectology.**](https://pdfs.semanticscholar.org/4bd4/0ed75369e07756b338f81a9c9529e207e279.pdf) In: *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, 26—31 May 2014, Reykjavik. 288-294.

Renfrew, C. and Heggarty, P. (2009). **Languages and Origins in Europe.** URL: http://www.languagesandpeoples.com/.

Heggarty, P. **Sound Comparisons: Exploring Phonetic Diversity Across Language Families** http://www.soundcomparisons.com/