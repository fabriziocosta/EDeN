EDeN
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



Examples
========


A few examples can be found as IPython Notebook inside of the examples folder.

* [Graph format](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/graph_format.ipynb)

* [Annotation](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/annotation.ipynb)

* [Classification](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/classification.ipynb)

* [General introduction](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/Sequence_example.ipynb)

* [RNA visualization and characterization](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA.ipynb)

* [RNA genome scan](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_genome_scan.ipynb)

* [RNA composite modelling](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_list.ipynb)

* [RNA genome scan with composite model and motif regex](http://nbviewer.ipython.org/github/fabriziocosta/EDeN/blob/master/examples/RNA_genome_scan_complex.ipynb)
