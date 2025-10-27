import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

import train
import eval

train_params = {}
'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds', 'iucn_taxa'
'''
train_params['species_set'] = 'iucn_taxa'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = 100

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
'''
train_params['input_enc'] = 'sin_cos'

'''
loss
- Which loss to use for training.
- Valid values: 
'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me',
 'informed_ssdl', 'informed_slds', 
 'hybrid_ssdl', 'hybrid_slds',
'contrastive1_ssdl', 'contrastive2_ssdl', 'contrastive1_slds', 'contrastive2_slds', 
'contrastive_full'
'''
train_params['loss'] = 'informed_ssdl'

'''
from where are pseudo-negatives being considered (needed for INFORMED LOSSES)
Values: 'absence', 'proximity', 'iucn's
'''
train_params['negatives_source'] = 'proximity' 

'''
Weighted loss
- Valid values: True, False
'''
train_params['weighted'] = True

'''
Policy for the probability of selecting between absence and proximity negative 
(needed for hybrid and contrastive1 losses)
- Valid values: 'constant', 'epoch'
'''
train_params['p_absence'] = 'constant'


def generate_experiment_name(params):
    experiment_name = (
        f"{params['hard_cap_num_per_class']}_"
        f"{params['loss']}"
    )
    if 'informed' in params['loss']:
        experiment_name += f"_{params['negatives_source']}"
    if params['weighted']:
        experiment_name += "_weighted"
    experiment_name += f"_{params['learning']}"
    experiment_name += f"_{params['species_set']}"
    return experiment_name


train_params = {
    'species_set': 'iucn_taxa',
    'hard_cap_num_per_class': 100,
    'num_aux_species': 0,
    'input_enc': 'sin_cos',
    'loss': 'hybrid_smart_full',
    'negatives_source': 'absence',
    'weighted': False,
    'learning': 'constant'
}
# train:
train.launch_training_run(train_params)


# evaluate:
for eval_type in ['iucn']: #, 'snt', 'iucn' , 'geo_prior', 'geo_feature'   CHANGE
    eval_params = {}
    eval_params['exp_base'] = './experiments'
    eval_params['experiment_name'] = train_params['experiment_name']
    eval_params['split'] = 'all'
    eval_params['eval_type'] = eval_type
    if eval_type == 'iucn':
        eval_params['device'] = torch.device('cpu') # for memory reasons
    cur_results = eval.launch_eval_run(eval_params)
    np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f"results_{eval_type}_{eval_params['split']}.npy"), cur_results)
    # Plot histogram AP per species
    df = pd.DataFrame({'taxa':cur_results['per_species_taxa_ids'], 'average_precission':cur_results['per_species_average_precision_all']})
    plt.hist(df['average_precission'])
    plt.xlabel("Average Precission")
    plt.ylabel("Number of Species")
    plt.axvline(np.nanmean(df['average_precission']), color = 'r', linestyle = '--', label='Mean AP')
    plt.legend()
    plt.savefig(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'ap_histogram_{eval_type}.png'))
    plt.close()

'''
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
'''


# & C:/Users/Diego/anaconda3/envs/sinr_icml/python.exe