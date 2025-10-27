import numpy as np
import pandas as pd
import random
import torch
import time
import os
import json
import h3


import setup
import utils
import models
import datasets

class EvaluatorSNT:
    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        self.loc_indices_per_species = D['loc_indices_per_species']
        self.labels_per_species = D['labels_per_species']
        self.taxa = D['taxa']
        self.obs_locs = D['obs_locs']
        self.obs_locs_idx = D['obs_locs_idx']

    def get_labels(self, species):
        species = str(species)
        lat = []
        lon = []
        gt = []
        for hx in self.data:
            cur_lat, cur_lon = h3.h3_to_geo(hx)
            if species in self.data[hx]:
                cur_label = int(len(self.data[hx][species]) > 0)
                gt.append(cur_label)
                lat.append(cur_lat)
                lon.append(cur_lon)
        lat = np.array(lat).astype(np.float32)
        lon = np.array(lon).astype(np.float32)
        obs_locs = np.vstack((lon, lat)).T
        gt = np.array(gt).astype(np.float32)
        return obs_locs, gt

    def run_evaluation(self, model, enc):
        results = {}

        # set seeds:
        np.random.seed(self.eval_params['seed'])
        random.seed(self.eval_params['seed'])

        # evaluate the geo model for each taxon
        results['per_species_average_precision_all'] = np.zeros((len(self.taxa)), dtype=np.float32)
        results['per_species_tpr'] = np.zeros((len(self.taxa)), dtype=np.float32)
        results['per_species_tnr'] = np.zeros((len(self.taxa)), dtype=np.float32)
        results['per_species_taxa_ids'] = np.array(self.taxa)  # ADDED TO INCLUDE TAXA ON PER SPECIES PRECISSION

        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        # generate model predictions for classes of interest at eval locations
        with torch.no_grad():
            loc_emb = model(loc_feat, return_feats=True)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1)).cpu().numpy()

        split_rng = np.random.default_rng(self.eval_params['split_seed'])
        for tt_id, tt in enumerate(self.taxa):

            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0:
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
                results['per_species_tpr'][tt_id] = np.nan
                results['per_species_tnr'][tt_id] = np.nan
            else:
                # generate ground truth labels for current taxa
                cur_loc_indices = np.array(self.loc_indices_per_species[tt_id])
                cur_labels = np.array(self.labels_per_species[tt_id])

                # apply per-species split:
                assert self.eval_params['split'] in ['all', 'val', 'test']
                if self.eval_params['split'] != 'all':
                    num_val = np.floor(len(cur_labels) * self.eval_params['val_frac']).astype(int)
                    idx_rand = split_rng.permutation(len(cur_labels))
                    if self.eval_params['split'] == 'val':
                        idx_sel = idx_rand[:num_val]
                    elif self.eval_params['split'] == 'test':
                        idx_sel = idx_rand[num_val:]
                    cur_loc_indices = cur_loc_indices[idx_sel]
                    cur_labels = cur_labels[idx_sel]

                # extract model predictions for current taxa from prediction matrix
                pred = pred_mtx[cur_loc_indices, tt_id]

                # compute the AP for each taxa
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_faster((cur_labels > 0).astype(np.int32), pred)

                # compute the FP and FN rate for each taxa
                results['per_species_tpr'][tt_id] , results['per_species_tnr'][tt_id] = utils.analyze_fp_fn((cur_labels > 0).astype(np.int32), pred, threshold=-2)

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)

        return results

    
    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')
    


class EvaluatorIUCN:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            self.data = json.load(f)
        self.obs_locs = np.array(self.data['locs'], dtype=np.float32)
        self.taxa = [int(tt) for tt in self.data['taxa_presence'].keys()]

    def run_evaluation(self, model, enc):

        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        # Per-species split logic
        split_rng = np.random.default_rng(self.eval_params['split_seed'])

        # Split species
        all_taxa = self.taxa
        split_type = self.eval_params['split']
        print(split_type)
        val_frac = self.eval_params['val_frac']

        # subset data based on val/test split
        if split_type in ['val', 'test']:
            shuffled_taxa = split_rng.permutation(all_taxa)
            num_val_species = int(len(all_taxa) * val_frac)
            if split_type == 'val':
                taxa_subset = set(shuffled_taxa[:num_val_species])
            else:  # 'test'
                taxa_subset = set(shuffled_taxa[num_val_species:])
        else:
            taxa_subset = set(all_taxa)
        taxa_subset = sorted(list(taxa_subset))
        print(len(taxa_subset))

        # classes subset
        classes_subset = []
        for t in taxa_subset:
            class_idx = np.where(np.array(self.train_params['class_to_taxa']) == t)[0]
            if len(class_idx) > 0:
                classes_subset.append(class_idx[0])

        # generate model predictions for classes of interest at eval locations
        with torch.no_grad():
            loc_emb = utils.batched_forward(model, loc_feat, batch_size=1024)
            wt = model.class_emb.weight[classes_subset, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1)).cpu().numpy()

        # Create results dictionary
        results = {}
        results['per_species_average_precision_all'] = np.zeros(len(taxa_subset), dtype=np.float32)
        results['per_species_precision'] = np.zeros(len(taxa_subset), dtype=np.float32)
        results['per_species_recall'] = np.zeros(len(taxa_subset), dtype=np.float32)
        results['per_species_threshold'] = np.zeros(len(taxa_subset), dtype=np.float32)
        results['per_species_taxa_ids'] = np.array(taxa_subset)


        # evaluation
        for i, tt in enumerate(taxa_subset):

            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0:
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][i] = np.nan
            else:
                # extract model predictions for current taxa from prediction matrix
                pred = pred_mtx[:, i]
                gt = np.zeros(obs_locs.shape[0], dtype=np.float32)
                gt[self.data['taxa_presence'][str(tt)]] = 1.0
                
                # average precision score:
                results['per_species_average_precision_all'][i] = utils.average_precision_score_faster(gt, pred)
                # compute metrics per species
                results['per_species_precision'][i], results['per_species_recall'][i], results['per_species_threshold'][i]   = utils.binary_metrics(gt, pred)


        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        per_species_precision = results['per_species_precision'][valid_taxa]
        per_species_recall = results['per_species_recall'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['mean_precision'] = per_species_precision.mean()
        results['mean_recall'] = per_species_recall.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(taxa_subset)
        return results

    
    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')
    

def launch_eval_run(overrides):

    eval_params = setup.get_default_params_eval(overrides)

    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    model.eval()

    # create input encoder:
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    print('\n' + eval_params['eval_type'])
    t = time.time()
    if eval_params['eval_type'] == 'snt':
        eval_params['split'] = 'test' # val, test, all
        eval_params['val_frac'] = 0.10
        eval_params['split_seed'] = 7499
        evaluator = EvaluatorSNT(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'iucn':
        eval_params.setdefault('split', 'all')
        eval_params.setdefault('val_frac', 0.60)
        eval_params.setdefault('split_seed', 7499)
        evaluator = EvaluatorIUCN(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    else:
        raise NotImplementedError('Eval type not implemented.')
    print(f'evaluation completed in {np.around((time.time()-t)/60, 1)} min')


    return results
