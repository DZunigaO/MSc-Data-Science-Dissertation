import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import h3
from collections import defaultdict
from scipy import stats

import torch
import utils
import gen_negative_candidates

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, class_to_taxa, classes,train_data, presence_info, candidates, input_enc, device):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster)

        # define some properties:
        self.train_data = train_data
        self.h3_count = presence_info['h3_count']
        self.presence_h3_cells = presence_info['presence_h3_cells']
        self.p_abs = presence_info['p_abs']
        self.n_cells = presence_info['n_cells']
        self.kl = presence_info['kl']
        self.candidates = candidates
        self.input_enc = input_enc
        self.device = device
        self.locs = train_data['locs']
        self.labels = train_data['labels']
        self.loc_feats = self.enc.encode(self.locs)
        self.input_dim = self.locs.shape[1]
        self.num_classes = len(class_to_taxa)
        self.class_to_taxa = class_to_taxa
        self.classes = classes

        # useful numbers:
        self.num_classes = len(self.class_to_taxa)
        self.input_dim = self.loc_feats.shape[1]

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        class_id  = self.labels[index]
        return loc_feat, loc, class_id
    


def load_env():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    return raster

def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

def load_inat_data(ip_file, taxa_of_interest=None):

    print('\nLoading  ' + ip_file)
    data = pd.read_csv(ip_file)

    # remove outliers
    num_obs = data.shape[0]
    data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
    if (num_obs - data.shape[0]) > 0:
        print(num_obs - data.shape[0], 'items filtered due to invalid locations')

    if 'accuracy' in data.columns:
        data.drop(['accuracy'], axis=1, inplace=True)

    if 'positional_accuracy' in data.columns:
        data.drop(['positional_accuracy'], axis=1, inplace=True)

    if 'geoprivacy' in data.columns:
        data.drop(['geoprivacy'], axis=1, inplace=True)

    if 'observed_on' in data.columns:
        data.rename(columns = {'observed_on':'date'}, inplace=True)

    num_obs_orig = data.shape[0]
    data = data.dropna()
    size_diff = num_obs_orig - data.shape[0]
    if size_diff > 0:
        print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

    # keep only taxa of interest:
    if taxa_of_interest is not None:
        num_obs_orig = data.shape[0]
        data = data[data['taxon_id'].isin(taxa_of_interest)]
        print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

    print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

    locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
    taxa = data['taxon_id'].values.astype(np.int64)

    if 'user_id' in data.columns:
        users = data['user_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    elif 'observer_id' in data.columns:
        users = data['observer_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    else:
        users = np.ones(taxa.shape[0], dtype=np.int64)*-1

    # Note - assumes that dates are in format YYYY-MM-DD
    years  = np.array([int(d_str[:4])   for d_str in data['date'].values])
    months = np.array([int(d_str[5:7])  for d_str in data['date'].values])
    days   = np.array([int(d_str[8:10]) for d_str in data['date'].values])
    days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
    dates  = days_per_month[months-1] + days-1
    dates  = np.round((dates) / 364.0, 4).astype(np.float32)
    if 'id' in data.columns:
        obs_ids = data['id'].values
    elif 'observation_uuid' in data.columns:
        obs_ids = data['observation_uuid'].values

    return locs, taxa, users, dates, years, obs_ids

def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_subset=None):
    print(f"species_set received: {species_set}, type: {type(species_set)}")  # Debug
    if species_set == 'all':
        return None
    if species_set in ['snt_birds', 'iucn_taxa']:
        assert taxa_file_subset is not None, f"taxa_file_subset is None for species_set {species_set}"
        with open(taxa_file_subset, 'r') as f:
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets[species_set])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123):
    if hard_cap == -1:
        return np.arange(len(labels))
    #print(f'subsampling (up to) {hard_cap} per class for the training set')
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

def get_h3_count(locs, resolution=1):
    counts = defaultdict(int)
    for lon, lat in locs:
        h3_cell = h3.geo_to_h3(lat, lon, resolution)
        counts[h3_cell] += 1
    return dict(counts)

def get_presence_info(labels, locs, resolution=1):
    labels_list = [l.item() if hasattr(l, "item") else l for l in labels]
    locs_list = [(float(lon), float(lat)) for lon, lat in locs]
    # Dictionary
    cell_counts = defaultdict(int) # to count number of observations per cell (used in location weights)
    dic_counts = defaultdict(lambda: defaultdict(int)) # to count number of observations per species-cell
    for label, (lon, lat) in zip(labels_list, locs_list):
        h3_cell = h3.geo_to_h3(lat, lon, resolution)
        cell_counts[h3_cell] += 1
        dic_counts[label][h3_cell] += 1
    # Prepare consistent indexing
    unique_species = sorted(dic_counts.keys())
    h3_cells = sorted(cell_counts.keys())
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    cell_to_idx = {c: i for i, c in enumerate(h3_cells)}
    # Build per species distribution matrix
    species_dist = np.zeros((len(unique_species), len(h3_cells)))
    for species, cell_counts in dic_counts.items():
        for cell, count in cell_counts.items():
            species_dist[species_to_idx[species]][cell_to_idx[cell]] = count
    # Normalize per species (species range distribution)
    row_sums = species_dist.sum(axis=1)
    species_dist = species_dist / row_sums[:, np.newaxis]
    # Normalize distribution of full data (average range distribution)
    total_dist = species_dist.sum(axis=0)
    total_dist = total_dist / total_dist.sum()
    # Get H3 count dictionary
    h3_count = dict(cell_counts)
    return species_dist, total_dist, h3_count


def get_positive_ssdl_candidates(class_to_taxa, class_ids, locs, output_path=None):
    pos_dict = {}
    # Get all locations and labels from the dataset
    locs_all = locs.cpu().numpy()  # Shape: (num_samples, 2)
    labels_all = class_ids.cpu().numpy()  # Shape: (num_samples,)
    # Build mapping from class_id to coordinates
    for class_id in np.unique(class_ids.cpu().numpy()):
        taxon_id = class_to_taxa[int(class_id)]
        matching = locs_all[labels_all == class_id]
        if len(matching) > 0:
            norm_coords = [[lon / 180.0, lat / 90.0] for lon, lat in matching] # normalize coordinates
            pos_dict[str(taxon_id)] = norm_coords
    # Save as JSON
    if output_path:
        with open(output_path, "w") as f:
            json.dump(pos_dict, f)
    print("Positive candidates extracted")
    return pos_dict

def get_positive_slds_candidates(cells_dict, output_path=None):
    pos_dict = gen_negative_candidates.invert_species_to_cells(cells_dict)
    # Save as JSON
    if output_path:
        with open(output_path, "w") as f:
            json.dump(pos_dict, f)
    return pos_dict


def get_train_data(params): 
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    obs_file  = os.path.join(data_dir, params['obs_file'])
    taxa_file = os.path.join(data_dir, params['taxa_file'])
    taxa_file_subset = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = get_taxa_of_interest(
        params['species_set'],
        params['num_aux_species'], 
        params['aux_species_seed'], 
        params['taxa_file'], taxa_file_subset
        )

    locs, labels, _, _, _, _ = load_inat_data(obs_file, taxa_of_interest)
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    # load class names
    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    # Compute and load negative candidates lists
    gen_negative_candidates.generate_candidates(
        data_dir = data_dir,
        locs=locs,
        class_ids=class_ids,
        class_to_taxa=class_to_taxa,
        h3_resolution=params['h3_resolution'],
        proximity_k=params['proximity_k']
        )
    
    with open(os.path.join(data_dir, 'presence_h3_cells.json'), 'r') as f:
        presence_h3_cells = json.load(f)
    with open(os.path.join(data_dir, 'absence_ssdl_candidates.json'), 'r') as f:
        absence_ssdl_candidates = json.load(f)
    with open(os.path.join(data_dir, 'proximity_ssdl_candidates.json'), 'r') as f:
        proximity_ssdl_candidates = json.load(f)
    with open(os.path.join(data_dir, 'absence_slds_candidates.json'), 'r') as f:
        absence_slds_candidates = json.load(f)
    with open(os.path.join(data_dir, 'proximity_slds_candidates.json'), 'r') as f:
        proximity_slds_candidates = json.load(f)

    # Load precomputed IUCN negative candidates
    with open(os.path.join(data_dir, 'iucn_absence_candidates.json'), 'r') as f:
        iucn_absence_candidates = json.load(f)

    # Subset to max number per species
    idx_ss = get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'])
    
    # get locs and labels
    locs = torch.from_numpy(np.array(locs)[idx_ss]) # convert to Tensor
    labels = torch.from_numpy(np.array(class_ids)[idx_ss])

    # Generate and load list with positive candidates
    positive_ssdl_candidates = get_positive_ssdl_candidates(class_to_taxa, labels, locs, output_path=os.path.join(data_dir, 'positive_ssdl_candidates.json'))
    positive_slds_candidates = get_positive_slds_candidates(presence_h3_cells, output_path=os.path.join(data_dir, 'positive_slds_candidates.json'))

    # Get species and cells counts and distributions
    species_dist, total_dist, h3_count = get_presence_info(labels, locs, resolution=params['h3_resolution'])

    # Calculate Preference fo Absence per species
    # Species range
    n_cells  = {k: len(v) for k, v in presence_h3_cells.items()}
    with open(os.path.join(data_dir, 'n_cells_species.json'), 'w') as f:
        json.dump(n_cells, f)
    # KL per species
    kl_divergence = np.array([stats.entropy(species_dist[i], total_dist) for i in range(len(unique_taxa))])
    kl_dict = {str(taxa): float(kl) for taxa, kl in zip(unique_taxa, kl_divergence)}
    with open(os.path.join(data_dir, 'kl_divergence.json'), 'w') as f:
        json.dump(kl_dict, f)
    """
    # Load Diff AP (Absence - Proximity) if saved
    file_path = os.path.join(data_dir, "ap_diff.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            ap_diff = json.load(f)
    else:
        print(f"File does not exist at: {file_path}")
    """
    # Preference for absence negative per species
    n_cells_array = np.array([n_cells[str(int(taxa))] for taxa in unique_taxa], dtype=float)
    p_abs = utils.get_p_absence(n_cells_array, kl_divergence)
    p_abs_dict = {str(taxa): float(kl) for taxa, kl in zip(unique_taxa, p_abs)}
    with open(os.path.join(data_dir, 'p_abs.json'), 'w') as f:
        json.dump(p_abs_dict, f)

    
    # effcient storing
    train_data = {
        'locs': locs,
        'labels': labels,
    }

    presence_info = {
        'h3_count' : h3_count,
        'presence_h3_cells' : presence_h3_cells,
        'p_abs' : p_abs_dict,
        'n_cells': n_cells,
        'kl': kl_dict,
    }
    
    candidates = {
        'absence_ssdl': absence_ssdl_candidates,
        'proximity_ssdl': proximity_ssdl_candidates,
        'absence_slds': absence_slds_candidates,
        'proximity_slds': proximity_slds_candidates,
        'positive_ssdl': positive_ssdl_candidates,
        'positive_slds': positive_slds_candidates,
        'iucn_absence_candidates': iucn_absence_candidates, # REMOVE
    }

    ds = LocationDataset(
        class_to_taxa, 
        classes,
        train_data,
        presence_info,
        candidates, 
        params['input_enc'], 
        params['device']
        )

    return ds
