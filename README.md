# # MSc Data Science Dissertation - Investigating the Role of Negative Data Selection in Deep Species Distribution Models

Code for training and evaluating global-scale species range estimation models.

## 🌍 Overview 
Estimating the geographical range of a species from sparse observations is a challenging and important geospatial prediction problem. Given a set of locations where a species has been observed, the goal is to build a model to predict whether the species is present or absent at any location. In this work, we use Spatial Implicit Neural Representations (SINRs) proposed by [Spatial Implicit Neural Representations for Global-Scale Species Mapping](https://arxiv.org/abs/2306.02564), to jointly estimate the geographical range of thousands of species simultaneously. We proposed the use of pseudo-absences based on excluding areas where species have already been observed (absence informed negatives) and pseudo-absences selected within the proximity of the observed area (proximity informed negatives). Showing that included both of them provide complementary information that outperforms a stragey of randomly selecting negatives across the world.

## 🔍 Getting Started 

#### Installing Required Packages

1. We recommend using an isolated Python environment to avoid dependency issues. Install the Anaconda Python 3.9 distribution for your operating system from [here](https://www.anaconda.com/download). 

2. Create a new environment and activate it:
```bash
 conda create -y --name sinr_icml python==3.9
 conda activate sinr_icml
```

3. After activating the environment, install the required packages:
```bash
 pip3 install -r requirements.txt
```


## 🗺️ Generating Predictions
To generate predictions for a model and compare them visually against the presence-absence from IUCN, run the following command: 
```bash
 python viz_prediction.py --taxa_id 130714 --model_path './experiments/100_hybrid_full_constant_iucn_taxa/model.pt' 
```
or specifying a different experiment path

Here, `--taxa_id` is the id number for a species of interest from [iNaturalist](https://www.inaturalist.org/taxa/130714). If you want to generate predictions for a random species, add the `--rand_taxa` instead. 

The model for the Hybrid Loss, our best experiment, are included inside the `experiments` folder

## 🚅 Training and Evaluating Models

To train and evaluate a model, run the following command:
```bash
 python train_and_evaluate_models.py
```

#### Hyperparameters
Common parameters of interest can be set within `train_and_evaluate_models.py`. All other parameters are exposed in `setup.py`. 

#### Outputs
By default, trained models and evaluation results will be saved to a folder in the `experiments` directory. Evaluation results will also be printed to the command line. 


## 📜 Disclaimer
Extreme care should be taken before making any decisions based on the outputs of models presented here. Our goal in this work is to demonstrate the promise of large-scale representation learning for species range estimation, not to provide definitive range maps. Our models are trained on biased data and have not been calibrated or validated beyond the experiments illustrated in the paper. 

