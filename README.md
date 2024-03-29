# Multiscale Atlas of Gene Expression for Integrative Cortical Cartography

Welcome to the MAGICC! This repository contains the data and code used to visualise and analyse vertex-level maps of cortical gene expression.

## Overview
The atlas was generated using [the Allen Human Brain Atlas](https://human.brain-map.org/). To create a dense transcriptomic atlas of the cortex, we used AHBA microarray measures of gene expression for 20,781 genes in each of 1304 cortical samples from six donor left cortical hemispheres. We extracted a model of each donor's cortical sheet by processing their [brain MRI scan](https:10.1016/j.neuroimage.2017.12.060), and identified the nearest cortical vertex of each postmortem cortical sample in this sheet. For each gene, we then propagated measured expression values into neighboring vertices using nearest-neighbor interpolation followed by smoothing. Expression values were scaled across vertices and these vertex-level expression maps were averaged across donors to yield a single dense expression map (DEM) for each gene - which provided estimates of expression at ~ 30,000 vertices across the cortical sheet (e.g. DEM for PVALB). These fine-grained vertex-level expression measures also enabled us to estimate the orientation and magnitude of expression change for each gene at every vertex.
Other fantastic AHBA resources are available including [abagen](https://abagen.readthedocs.io/en/stable/) and [Gregor Gryglewski's atlas](http://www.meduniwien.ac.at/neuroimaging/mRNA.html)

![Methods for creating vertex-level expression maps](imgs/methods.png)


## Data and Code
The full vertex-level dataset can be downloaded from [here](https://rdr.ucl.ac.uk/articles/dataset/MAGICC_vertex-level_gene_expression_data/22183891/1). These data include processed gene expression data, as well as cortical surfaces for visualisation. The code includes pipelines for downloading data and visualising results.

## Interactive Usage
To look at a cortical expression map for a gene of interest, you can try [this interactive collaborative notebook](https://colab.research.google.com/github/kwagstyl/MAGICC/blob/main/notebooks/gene2map.ipynb). The interface allows users to plot their expression and gradient.

## Installation and Usage
To download and analyse the data on your own computer, take a look at the notebooks capturing example analyses from our paper.


## Citation
If you use the Multiscale Atlas of Gene Expression for Integrative Cortical Cartography in your research, please cite our paper:

[Transcriptional Cartography Integrates Multiscale Biology of the Human Cortex](https://doi.org/10.7554/eLife.86933.1)

## Contact
If you have any questions or comments about the atlas or this repository, feel free to contact me at k.wagstyl@ucl.ac.uk. We welcome feedback and suggestions for how to improve the project.
