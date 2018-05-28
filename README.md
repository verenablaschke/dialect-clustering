## Current Data

The BDPA contains a list of 111 cognate sets across 21 German/Dutch dialects, transcribed in IPA, tokenized (affricates and diphthongs constitute single segments), and already aligned. (More details in the ```data``` folder.) All of the entries were transcribed by a single person and aligned by another.

### TODO

- Unfortunately, the BDPA contains only two Low German dialects. Check if I can add some of the ones from [Sound Comparisons (Heggarty)](http://www.soundcomparisons.com/#/en/Germanic) (heggartynodatesound).
  - Were they also transcribed by the same person?
  - They aren't aligned with the other entries.
    - It would be possible however to start the program with an alignment step whose success is measured with the gold standard data.
    - If I add just one or two dialects, manual alignment might be worthwhile.
- Possibly also check to what extend I might be able to use data from the Alignments of the Phonetischer Atlas von Deutschland (PAD) (prokic2018alignments), the Indo-European Lexical Cognacy Database (IELex) (dunn2012indo-european), the Morfologische Atlas van de Nederlandse Dialecten (MAND) (goemannodatemorfologische), or the Dynamische Syntactische Atlas van de Nederlandse Dialecten (DynaSAND) (barbiersnodatedynamische). Of course, for all of these, the transcription standards would probably differ from what I currently use.
- Maybe check if the non-IPA phonetic transcriptions of the Norwegian and Swedish parts of the [Nordic Dialect Corpus](http://www.tekstlab.uio.no/nota/scandiasyn/) are compatible with one another.

## Current implementation 

[extract_features.py](https://github.com/verenablaschke/dialect-clustering/blob/master/extract_features.py) currently contains a very rough implementation that does the following:
- For each dialect, concatenate all entries (including gap segments from the alignment) and convert the IPA segments into phonetic feature vectors using PanPhon, resulting in one very long feature vector per dialect.
- Get the Manhattan distance between each pair of dialects to create a distance matrix (visualized with the heatmap).
- Perform hierarchical clustering on the feature vectors using SciPiy's implementation of the UPGMA algorithm (Unweighted Pair Group Method using Arithmetic averages) (visualized with the dendrogram).

What is interesting is that of the (currently only 2...) Low German dialects, one is grouped with the Dutch variants and the other is part of an otherwise High & Central German cluster.

### TODO

- Improve the feature vector conversion & distance measure:
  [ ] Deal with diphthongs.
  [x] If two dialects share a gap segment in an aligned entry, don't let that facture into the distance score. Changed the scores in the distance matrix, but didn't change the dendrogram.
  - Appropriate gap penalization.
- Try out different distance measures and clustering algorithms:
  - heeringa2006evaluation present and evaluate a bunch of different alignment/distance scoring strategies, such as including n-grams for phonetic context. See also nerbonne1997measuring.
  - heeringa2004measuring (ch. 6.1.3) compares different clustering algorithms (and ultimately prefers UPGMA)
  - multi-dimensional scaling (nerbonne2009data-driven, heeringa2004measuring ch. 6.2) and other dimensionality-reduction techniques prior to clustering
  - There are implementations where the focus is also on the features that distinguish dialect groups (prokic2012detecting, nerbonne2006identifying, wieling2011bipartite)
- Figure out why LingPy crashes when trying to create a dendrogram from the data.
- All of this is about the number of differences. It would also be interesting to consider the number of regular correspondences vs. unpredicatable correspondences/differences.

## Evaluation

### TODO

- Literature research on (non-statistical) analyses of the German/Dutch dialect landscape. Get some hierarchy that I can compare my tree to?

# References

## Libraries

PanPhon v. 0.14 ([GitHub](https://github.com/dmort27/panphon), [PyPi](https://pypi.org/project/panphon/)):

- Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L. (2016). [**PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors.**](https://www.aclweb.org/anthology/C/C16/C16-1328.pdf) In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers* (pp. 3475-3484).

SciPy (cluster, distance)

## Data

List, J.-M. and Prokić, J. (2014). [**A benchmark database of phonetic alignments in historical linguistics and dialectology.**](https://pdfs.semanticscholar.org/4bd4/0ed75369e07756b338f81a9c9529e207e279.pdf) In: *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, 26—31 May 2014, Reykjavik. 288-294.

Renfrew, C. and Heggarty, P. (2009). **Languages and Origins in Europe.** URL: http://www.languagesandpeoples.com/.