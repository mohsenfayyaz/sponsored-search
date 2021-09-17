# sponsored-search
This repository contains implementation of a simple sponsored search advertising model 
for generating desired embeddings from queries and ads in the same representation space.

## Pre-requisite Installation
### With pip
```bash
pip install transformers
```
 ### With conda
```shell script
conda install -c huggingface transformers
```
 
 ## Running
Sample notebook is available in [query_ad.ipynb](query_ad.ipynb)
 
 ## Implementation Process
* Study the [paper](https://arxiv.org/pdf/1607.01869.pdf) and related material
* Create negative sampled dataset from the raw dataset
* Implement DatasetHandler for processing the data
* Implement Pytorch model in a multi-file manner
* Implement proper tests for ensuring integrity