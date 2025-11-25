# PAM_data_sampling
Scripts and data supporting the paper ["Data-driven Sampling Strategies for Fine-Tuning Bird Detection Models"](https://www.biorxiv.org/content/10.1101/2025.10.02.679964v1)

Authors: Corentin Bernard, Ben McEwen, Benjamin Cretois, Hervé Glotin, Dan Stowell, Ricard Marxer

This work is part of the [TABMON](https://tabmon-eu.nina.no/) project.

Annotated data comes from the [WABAD](https://zenodo.org/records/14191524) dataset. Only European sites have been retained.


### Getting Started

Create virtual environment and specify python version 3.13.2: `conda create --name tabmon python=3.13.2`.\
Install dependencies: `pip install -r requirements.txt`

### Run reverse correlation on the WABAD dataset

Reverse correlation parameters (number of samples and number of iterations) can be set in the `config.yaml` file.\
Then, run :
```
python run_reverse_correlation.py
```

### Analyze ceverse correlation results

Results can be vizaualized on the Jupyter Notebook: `analyze_reverse_correlation.ipynb`\



The repository will be updated soon with other scripts and pre-processed data.
