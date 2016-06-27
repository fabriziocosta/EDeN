

[![DOI](https://zenodo.org/badge/10054/fabriziocosta/EDeN.svg)](https://zenodo.org/badge/latestdoi/10054/fabriziocosta/EDeN)
[![Build Status](https://travis-ci.org/fabriziocosta/EDeN.svg?branch=master)](https://travis-ci.org/fabriziocosta/EDeN)

<p align="center"><img src="EDeN_logo.png"></p>

[EDeN](http://fabriziocosta.github.io/EDeN)
====

The Explicit Decomposition with Neighborhoods (EDeN) is a decompositional kernel based on the Neighborhood Subgraph Pairwise Distance Kernel (NSPDK) that can be used to induce an explicit feature representation for graphs. This in turn allows the adoption of machine learning algorithm to perform supervised and unsupervised learning task in a scalable way (e.g. using fast stochastic gradient descent methods in classification and approximate neighborhood queries in clustering).

Among the novelties introduced in EDeN is the ability to take in input real vector labels and to process weighted and nested graphs.



Examples
========


A few examples can be found as IPython Notebook at the following GitHub repository: [EDeN_examples](https://github.com/fabriziocosta/EDeN_examples).

 * [Introduction](http://nbviewer.ipython.org/github/fabriziocosta/EDeN_examples/blob/master/Sequence_example.ipynb)
 * [Graph format](http://nbviewer.ipython.org/github/fabriziocosta/EDeN_examples/blob/master/graph_format.ipynb)
 * [Classification](http://nbviewer.ipython.org/github/fabriziocosta/EDeN_examples/blob/master/classification.ipynb)
 * [Annotation](http://nbviewer.ipython.org/github/fabriziocosta/EDeN_examples/blob/master/annotation.ipynb)


Installation
============

You can install EDeN with pip directly from github.

```
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```

### Docker

You can create a docker image containing your versions of EDeN with all necessary dependencies installed using the command

```bash
docker build -t eden .
```

Running this image will start a Jupyter notebook server. You can access this server on your machine via [http://localhost:8080](http://localhost:8080). Using  option ```-v /home/user/EDeN/examples/:/export/``` will make the notebooks in your local folder ```/home/user/EDeN/examples/``` accessible to Jupyter.

```bash
docker run -i -p 8080:8888 -v /home/user/EDeN/examples/:/export/ eden
```

References
==========

Costa, Fabrizio, and Kurt De Grave. "Fast neighborhood subgraph pairwise distance kernel." Proceedings of the 26th International Conference on Machine Learning, 2010. ([ref](http://www.icml2010.org/papers/347.pdf))


K. De Grave, F. Costa, "Molecular Graph Augmentation with Rings and Functional Groups", Journal of Chemical Information and Modeling, 50 (9), pp 1660–1668, 2010. ([ref](http://pubs.acs.org/doi/abs/10.1021/ci9005035))

Steffen Heyne, Fabrizio Costa, Dominic Rose, and Rolf Backofen,"GraphClust: alignment-free structural clustering of local RNA secondary structures",Bioinformatics, 28 no. 12 pp. i224-i232, 2012.
([ref](http://bioinformatics.oxfordjournals.org/content/28/12/i224))


Kousik Kundu, Fabrizio Costa, and Rolf Backofen, "A graph kernel approach for alignment-free domain-peptide interaction prediction with an application to human SH3 domains", Bioinformatics, 29 no. 13 pp. i335-i343, 2013. ([ref](http://bioinformatics.oxfordjournals.org/content/29/13/i335))


P. Frasconi, F. Costa, K. De Grave, L. De Raedt,"kLog: A Language for Logical and Relational Learning with Kernels", Artificial Intelligence, 2014. ([ref](http://www.sciencedirect.com/science/article/pii/S0004370214001064))

Omer S. Alkhnbashi, Fabrizio Costa, Shiraz A. Shah, Roger A. Garrett, Sita J. Saunders and Rolf Backofen, "CRISPRstrand: Predicting repeat orientations to determine the crRNA-encoding strand at CRISPR loci", ECCB, 13th European Conference on Computational Biology, 2014. ([ref](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4147912/))

Videm P., Rose D., Costa F., Backofen R. ,"BlockClust: efficient clustering and classification of non-coding RNAs from short read RNA-seq profiles", Bioinformatics, 2014 Jun 15;30(12):i274-82. ([ref](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4058930/))

Daniel Maticzka, Sita J Lange, Fabrizio Costa, Rolf Backofen, "GraphProt: modeling binding preferences of RNA-binding proteins", Genome Biology 2014, 15:R17 (22 January 2014). ([ref](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4053806/))

Gianluca Corrado, Toma Tebaldi, Giulio Bertamini, Fabrizio Costa, Alessandro Quattrone, Gabriella Viero and Andrea Passerini, "PTRcombiner: mining combinatorial regulation of gene expression from post-transcriptional interaction maps", BMC Genomics, 2014; 15:304. ([ref](http://www.biomedcentral.com/1471-2164/15/304/abstract))

R. Ferrarese, G. R. 4th Harsh, A. K. Yadav, E. Bug, D. Maticzka, W. Reichardt, S. M. Dombrowski, T. E. Miller, A. P. Masilamani, F. Dai, H. Kim, M. Hadler, D. M. Scholtens, I. L. Y. Yu, J. Beck, V. Srinivasasainagendra, F. Costa, N. Baxan, D. Pfeifer, D. V. Elverfeldt, R. Backofen, A. Weyerbrock, C. W. Duarte, X. He, M. Prinz, J. P. Chandler, H. Vogel, A. Chakravarti, J. N. Rich, M. S. Carro, M. Bredel, "Lineage-specific splicing of a brain-enriched alternative exon promotes glioblastoma progression", J Clin Invest, 124 no. 7 pp. 2861-2876, 2014. ([ref](http://www.jci.org/articles/view/68836))

