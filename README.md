# PAM_data_sampling
Scripts and data supporting the paper ["Data-driven Sampling Strategies for Fine-Tuning Bird Detection Models"](https://www.biorxiv.org/content/10.1101/2025.10.02.679964v1)

Authors: Corentin Bernard, Ben McEwen, Benjamin Cretois, Hervé Glotin, Dan Stowell, Ricard Marxer.

This work is part of the [TABMON](https://tabmon-eu.nina.no/) project.

Annotated data comes from the [WABAD](https://zenodo.org/records/14191524) dataset. Only European sites have been retained.
Pre-processed data in 'dataset\' contains WABAD data cut into 3 sec samples and split into train, validation and test.
The .pkl files contain birdNET embeddings (x), and true labels (y), as well as dataframes with [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) confidence score, uncertainty (binary entropy of the predictions) and [acoustic indices](https://scikit-maad.github.io/).

### Getting Started

Create virtual environment and specify python version 3.13.2: `conda create --name revcor python=3.13.2`.\
Install dependencies: `pip install -r requirements.txt`

### Run reverse correlation on the WABAD dataset

Set parameters for reverse correlation in `config.yaml` : number of samples and number of iterations.\
Then, run :
```
python run_reverse_correlation.py
```

### Analyze reverse correlation results

Vizualize results on the Jupyter Notebook: 
```
analyze_reverse_correlation.ipynb
```

### Run evaluation of diverse sampling strategies on the WABAD dataset

Set parameters for reverse correlation in `config.yaml`: choice of sampling parameters (acoustic indices, birdNET confidence or uncertainty) and the number of iteration per condition.\
\
Then, run :
```
python run_sampling_evaluation.py
```

### Analyze sampling strategies evaluation

Vizualize results on the Jupyter Notebook: 
```
analyze_sampling_evaluation.ipynb
```


The repository will be updated soon with other scripts and pre-processed data.
