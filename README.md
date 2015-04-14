

[![DOI](https://zenodo.org/badge/10054/fabriziocosta/EDeN.svg)](http://dx.doi.org/10.5281/zenodo.15094)



[EDeN](http://fabriziocosta.github.io/EDeN)
====

The Explicit Decomposition with Neighborhoods (EDeN) is a decompositional kernel based on the Neighborhood Subgraph Pairwise Distance Kernel (NSPDK) that can be used to induce an explicit feature representation for graphs. This in turn allows the adoption of machine learning algorithm to perform supervised and unsupervised learning task in a scalable way (e.g. fast stochastic gradient descent methods in classification).

Among the novelties introduced in EDeN is the ability to take in input real vector labels and to process weighted graphs.


Installation
============

You can install EDeN with pip directly from github.

```python
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```

References
==========

Costa, Fabrizio, and Kurt De Grave. "Fast neighborhood subgraph pairwise distance kernel." Proceedings of the 26th International Conference on Machine Learning. 2010. ([PDF](http://www.icml2010.org/papers/347.pdf))

P. Frasconi, F. Costa, K. De Grave, L. De Raedt,"kLog: A Language for Logical and Relational Learning with Kernels", Artificial Intelligence, 2014. ([PDF](http://www.sciencedirect.com/science/article/pii/S0004370214001064)) 

Omer S. Alkhnbashi, Fabrizio Costa, Shiraz A. Shah, Roger A. Garrett, Sita J. Saunders and Rolf Backofen, "CRISPRstrand: Predicting repeat orientations to determine the crRNA-encoding strand at CRISPR loci", ECCB, 13th European Conference on Computational Biology, 2014. ([PDF](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4147912/))


Videm P., Rose D., Costa F., Backofen R. ,"BlockClust: efficient clustering and classification of non-coding RNAs from short read RNA-seq profiles", Bioinformatics. 2014 Jun 15;30(12):i274-82. ([PDF](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4058930/))

Examples
========


A few examples can be found as IPython Notebook inside of the examples folder.

* [Graph format](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/graph_format.ipynb)

* [Annotation](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/annotation.ipynb)

* [Classification](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/classification.ipynb)

* [General introduction](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/Sequence_example.ipynb)

* [RNA introduction](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_example.ipynb)

* [RNA visualization](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_visualization.ipynb)

* [RNA characterization](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA.ipynb)

* [RNA composite modelling](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_list.ipynb)

* [RNA genome scan](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_genome_scan.ipynb)
