# GISST

This repository contains the code for GISST (Graph Neural Networks Including Sparse Interpretability). This method provides important node feature and sub-graph edge interpretation in end-to-end training. Please read `example_gisst.ipynb` for a quick tutorial on how to use the APIs in this repository. 

GISST is developed by Chris Lin, Marylens Hernandez, Gerald Sun, Krishna Bulusu, and Jonathan Dry.

The name SIG was used during the development of this repository. The method was later renamed GISST (Graph Neural Networks Including Sparse Interpretability
). The preprint of GISST is available on [arXiv](https://arxiv.org/abs/2007.00119).

# Requirements

The developers are updating GISST for a stable version of PyTorch Geometric.

# Instructions

If using this repository for the first time, execute `setup.sh` to set up additional project sub-directories and preprocess real-world datasets (Mutagenicity, REDDIT-BINARY, and PROTEINS).

# Graph datasets

Synthetic datasets are included in `graphs`. After preprocessing through `setup.sh`, real-world datasets will also be added to your local `graphs` directory.

# Trained models

Models that have been tuned and trained are included in `output`. The directory of each trained model contains the following files.

* `configs.yaml`: configuration file with the hyperparameters that were searched through.
* `best_results.txt`: classification results of the trained model.
* `best_hyperparams.txt`: the optimal set of hyperparameters selected for the trained model.
* `best_state_dict.pt`: the parameters/weights of the trained model.

With `best_hyperparams.txt` and `best_state_dict.pt`, the trained model can be restored in Pytorch (see `explain_node.py` for example).