# ADMET properties prediction
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

- [ADMET properties prediction](#admet-properties-prediction)
  - [About](#about)
  - [Usage](#usage)
  - [App](#app)
  - [Citation](#citation)

`predictADMET` is a project to predict 5 different properties (membrane permeability, solubility, protein plasma binding, genotoxicity, cardiotoxicity) of molecules and get explanations (why a molecule is predicted to have a property).

## About 
Prediction of ADMET properties was done with the use of three types of ML regression models: 6 shallow models, 3 ensembles of shallow models, and 2 deep learning models, more precisely:
* Ridge Regression (`RR`),
* Random Forest Regressor (`RF`), 
* Histogram-based Gradient Boosting Regression Tree (`HistGrad`),
* Support Vector Regression (`SVR`),
* Extreme Gradient Boosting (`XGBoost`),
* Light Gradient Boosting Machine (`LGBM`),
* Ensemble models,
* Fully Connected Neural Networks (`FCNNs`),
* Graph Convolutional Neural Networks (`GCNNs`), 

and representations: `MACCSFp`, `PubchemFp`, `KRFp`, `molecular graphs`.

More details and results available at:

## Usage

Create conda environment and install packages:
```sh
conda create -n predictADMET python=3.8
conda activate predictADMET
pip install -r requirements.txt
```

Download data used in experiments from [link](https://drive.google.com/drive/u/0/folders/1NYHdDnOjMdqqBhDmRRRQT4mok3xtXUH2) or add your own data to the `data` directory.


Run training:
```sh
# scheme
python src/main.py -dataset dataset -data_type data_type -model model
# example
python src/main.py -dataset cardio -data_type klek -model rf
```

## App

All developed tools can be used via the online platform available at:

[![app](https://raw.githubusercontent.com/JamEwe/predictADMET/ADMET_prediction_app_screen.png)](https://admet.if-pan.krakow.pl)

## Citation

```bibtex
@Article{
}
```


