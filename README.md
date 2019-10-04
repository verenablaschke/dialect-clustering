# Clustering dialect varieties based on historical sound correspondences

Can we (meaningfully) cluster dialects based on sound correspondences?
Research such as [Wieling and Nerbonne (2011)](https://hal.archives-ouvertes.fr/hal-00730283/document)
uses phonetically transcribed doculect data that has been aligned with data from a reference doculect
to investigate clustering based on the presence/absence of sound segment alignments.
The results are doculect clusters as well as analyses of correlations between segment alignments and clusters.

Using data from [Heggarty (2018)](https://soundcomparisons.com/#/en/Germanic/),
I follow a similar approach, but use Proto-Germanic data as reference doculect
for a set of (continental) West Germanic doculects
to explore how historical sound shifts are associated with the resulting clusters.

Details on this are in my Bachelor's thesis, which can be found
[here](https://github.com/verenablaschke/dialect-clustering/blob/master/doc/Verena-Blaschke_BA-Thesis.pdf).
A [summary](https://github.com/verenablaschke/dialect-clustering/blob/master/doc/Verena-Blaschke_BA-Thesis_Summary.pdf) is also available. 

### Abstract

While information on historical sound shifts plays an important role for examining
the relationships between related language varieties,
it has rarely been used for computational dialectology.
This thesis explores the performance of two algorithms
for clustering language varieties
based on sound correspondences between Proto-Germanic
and modern continental West Germanic dialects.
Our experiments suggest that the results of agglomerative clustering
match common dialect groupings more closely
than the results of (divisive) bipartite spectral graph co-clustering.
We also observe that adding phonetic context information
to the sound correspondences yields clusters
that are more frequently associated with representative and distinctive
sound correspondences).

### Errata

The last sentence of section 4.3.2 _Bipartite Spectral Graph Co-clustering_ (p. 13) should read "The results from this method are hereafter referred to as ___BSGC__-context_ and ___BSGC__-nocontext_."
