import torch
from shapely.geometry import shape, Point
import h3
import numpy as np
import random

import utils

def get_loss_function(params):
    if params['loss'] == 'an_full':
        return an_full
    elif params['loss'] == 'an_slds':
        return an_slds
    elif params['loss'] == 'an_ssdl':
        return an_ssdl
    elif params['loss'] == 'an_full_me':
        return an_full_me
    elif params['loss'] == 'an_slds_me':
        return an_slds_me
    elif params['loss'] == 'an_ssdl_me':
        return an_ssdl_me
    elif params['loss'] == 'informed_ssdl':
        return informed_ssdl
    elif params['loss'] == 'informed_slds':
        return informed_slds
    elif params['loss'] == 'informed_full':
        return informed_full
    elif params['loss'] == 'dual_ssdl':
        return dual_ssdl
    elif params['loss'] == 'dual_slds':
        return dual_slds
    elif params['loss'] == 'dual_full':
        return dual_full
    elif params['loss'] == 'hybrid_ssdl':
        return hybrid_ssdl
    elif params['loss'] == 'hybrid_slds':
        return hybrid_slds
    elif params['loss'] == 'hybrid_full':
        return hybrid_full
    elif params['loss'] == 'dual_smart_ssdl':
        return dual_smart_ssdl
    elif params['loss'] == 'dual_smart_slds':
        return dual_smart_slds
    elif params['loss'] == 'dual_smart_full':
        return dual_smart_full
    elif params['loss'] == 'hybrid_smart_ssdl':
        return hybrid_smart_ssdl
    elif params['loss'] == 'hybrid_smart_slds':
        return hybrid_smart_slds
    elif params['loss'] == 'hybrid_smart_full':
        return hybrid_smart_full
    
    
    

    



def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    entropy = p * neg_log(p) + (1-p) * neg_log(1-p)
    return entropy


# ========== ASSSUMED NEGATIVE LOSSES ==========

def an_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights

    # data loss
    loss_pos = weights *neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_slds(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # get location embeddings
    loc_emb = model(loc_feat, return_feats=True)
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    
    # select random class (species)
    num_classes = loc_pred.shape[1]
    bg_class = torch.randint(low=0, high=num_classes-1, size=(batch_size,), device=params['device'])
    bg_class[bg_class >= class_id[:batch_size]] += 1

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights *  (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss


def an_full(batch, model, params, loc_to_feats, neg_type='hard'):
    
    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    # get predictions for locations and background locations
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log(1.0 - loc_pred) # assume negative
        loss_bg = neg_log(1.0 - loc_pred_rand) # assume negative
    elif neg_type == 'entropy':
        loss_pos = -1 * bernoulli_entropy(1.0 - loc_pred) # entropy
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand) # entropy
    else:
        raise NotImplementedError
    # Apply location weight
    weights = weights.unsqueeze(1)              
    loss_pos = loss_pos * weights
    loss_bg = loss_bg * weights
    # Apply positive labe weight (Cole et al.)
    loss_pos[inds[:batch_size], class_id] = params['pos_weight'] * neg_log(loc_pred[inds[:batch_size], class_id]) # Overrides loss_pos for true species
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss


def an_full_me(batch, model, params, loc_to_feats):

    return an_full(batch, model, params, loc_to_feats, neg_type='entropy')


def an_ssdl_me(batch, model, params, loc_to_feats):
    
    return an_ssdl(batch, model, params, loc_to_feats, neg_type='entropy')


def an_slds_me(batch, model, params, loc_to_feats):
    
    return an_slds(batch, model, params, loc_to_feats, neg_type='entropy')



# ========== INFORMED NEGATIVE LOSSES (one negative) ==========

def informed_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    if params['negatives_source'] == 'absence':
        neg_candidates = dataset.candidates['absence_ssdl']
    elif params['negatives_source'] == 'proximity':
        neg_candidates = dataset.candidates['proximity_ssdl']
    elif params['negatives_source'] == 'iucn':
        neg_candidates = dataset.iucn_absence_candidates
    else:
        print('Negative source not implemented')

    # draw randomly from negative candidates and extract features
    rand_loc = []
    class_to_taxa = dataset.class_to_taxa
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        neg_candidates_id = neg_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        rand_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    rand_loc = torch.cat(rand_loc, dim=0)
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss


def informed_slds(batch, model, params, loc_to_feats, neg_type='hard'):


    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    if params['negatives_source'] == 'absence':
        neg_candidates = dataset.candidates['absence_slds']
    elif params['negatives_source'] == 'proximity':
        neg_candidates = dataset.candidates['proximity_slds']
        neg_candidates_fallback = dataset.candidates['absence_slds']
    else:
        print('Negative source not implemented')

    # draw randomly from negative candidates and extract features
    bg_class = []
    class_to_taxa = dataset.class_to_taxa
    for loc in locs:
        # GET H3 CELL OF LOC
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # DRAW FROM NEGATIVE LIST OF H3 CELL
        candidate_taxa = neg_candidates[loc_cell] if loc_cell in neg_candidates else neg_candidates_fallback[loc_cell]
        taxa = random.choice(candidate_taxa)
        class_idx = class_to_taxa.index(int(taxa))
        bg_class.append(class_idx)
    bg_class = torch.tensor(bg_class, device=params['device'])

    # get location embeddings and predictions
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss


def informed_full(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    if params['negatives_source'] == 'absence':
        neg_candidates_ssdl = dataset.candidates['absence_ssdl']
        neg_candidates_slds = dataset.candidates['absence_slds']
    elif params['negatives_source'] == 'proximity':
        neg_candidates_ssdl = dataset.candidates['proximity_ssdl']
        neg_candidates_slds = dataset.candidates['proximity_slds']
        neg_candidates_slds_fallback = dataset.candidates['absence_slds'] #fallback
    else:
        print('Negative source not implemented')

    # ======= Negatives =======
    class_to_taxa = dataset.class_to_taxa
    # SSDL negative
    rand_loc = []
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        neg_candidates_id = neg_candidates_ssdl[taxon_id]
        loc = random.choice(neg_candidates_id)
        rand_loc.append(torch.tensor(loc).view(1, -1))
    rand_loc = torch.cat(rand_loc, dim=0).to(params['device'])
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    # SLDS negative
    neg_classes = []
    for loc in locs:
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        neg_candidates_id = neg_candidates_slds[loc_cell] if loc_cell in neg_candidates_slds else neg_candidates_slds_fallback[loc_cell]
        taxa = random.choice(neg_candidates_id)
        neg_classes.append(class_to_taxa.index(int(taxa)))
    neg_classes = torch.tensor(neg_classes, device=params['device'])
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_neg_ssdl = weights * neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id])
        loss_neg_slds = weights * neg_log(1.0 - loc_pred[inds[:batch_size], neg_classes])
    elif neg_type == 'entropy':
        loss_neg_ssdl = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id]))
        loss_neg_slds = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds, neg_classes]))
    else:
        raise NotImplementedError

    # Total loss
    loss = 2*loss_pos.mean() + loss_neg_ssdl.mean() + loss_neg_slds.mean()
    
    return loss


# ========== HYBRID INFORMED NEGATIVE LOSSES (changing negative source) ==========

def hybrid_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']

    # draw randomly from negative candidates and extract features
    rand_loc = []
    class_to_taxa = dataset.class_to_taxa
    # probability of drawing from absence
    if params['learning'] == 'constant':
        p = 0.5
    elif params['learning'] == 'epoch':
        p = max(0.0, 0.8 - params['epoch'] / 20)
    # Compute negatives
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        if np.random.rand() < p: # 0.5 chance of absence negative
            neg_candidates_id = absence_ssdl_candidates[taxon_id]
        else:   # 0.5 chance of proximity negative
            neg_candidates_id = proximity_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        rand_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    rand_loc = torch.cat(rand_loc, dim=0)
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def hybrid_slds(batch, model, params, loc_to_feats, neg_type='hard'):


    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # draw randomly from negative candidates and extract features
    bg_class = []
    class_to_taxa = dataset.class_to_taxa
    # probability of drawing from absence
    if params['learning'] == 'constant':
        p = 0.5
    elif params['learning'] == 'epoch':
        p = max(0.0, 0.8 - params['epoch'] / 20)
    for loc in locs:
        # GET H3 CELL OF LOC
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # DRAW FROM NEGATIVE LIST OF H3 CELL
        if np.random.rand() < p: # 0.5 chance of absence negative
            neg_candidates_id = absence_slds_candidates[loc_cell]
        else:   # 0.5 chance of proximity negative
            neg_candidates_id = proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        taxa = random.choice(neg_candidates_id)
        class_idx = class_to_taxa.index(int(taxa))
        bg_class.append(class_idx)
    bg_class = torch.tensor(bg_class, device=params['device'])

    # get location embeddings and predictions
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss


def hybrid_full(batch, model, params, loc_to_feats, neg_type='hard'):

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size, device=params['device'])

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']


    # ======= Negatives =======
    class_to_taxa = dataset.class_to_taxa
    # Probability of drawing from absence
    if params['learning'] == 'constant':
        p = 0.5
    elif params['learning'] == 'epoch':
        p = max(0.0, 1.0 - params['epoch'] / 10)
    rand_loc = []
    neg_classes = []
    for i in range(batch_size):
        # extract values
        class_id_val = class_id[i]
        taxon_id = str(class_to_taxa[class_id_val.item()])
        loc = locs[i]
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # select absence or proximity
        if np.random.rand() < p:
            neg_candidates_id_ssdl = absence_ssdl_candidates[taxon_id]
            neg_candidates_id_slds = absence_slds_candidates[loc_cell]
        else:
            neg_candidates_id_ssdl = proximity_ssdl_candidates[taxon_id]
            neg_candidates_id_slds = proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        # Location features for SSDL:
        loc = random.choice(neg_candidates_id_ssdl)
        rand_loc.append(torch.tensor(loc).view(1, -1))
         # Append background class for SLDS
        taxa = random.choice(neg_candidates_id_slds)
        neg_classes.append(class_to_taxa.index(int(taxa)))

    # Extract features background location
    rand_loc = torch.cat(rand_loc, dim=0).to(params['device'])
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # Background classes to tensor
    neg_classes = torch.tensor(neg_classes, device=params['device'])
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_neg_ssdl = weights * neg_log(1.0 - loc_pred_rand[inds, class_id])
        loss_neg_slds = weights * neg_log(1.0 - loc_pred[inds, neg_classes])
    elif neg_type == 'entropy':
        loss_neg_ssdl = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds, class_id]))
        loss_neg_slds = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds, neg_classes]))
    else:
        raise NotImplementedError

    # Total loss
    loss = 2*loss_pos.mean() + loss_neg_ssdl.mean() + loss_neg_slds.mean()
    
    return loss


# ========== DUAL INFORMED NEGATIVE LOSSES (changing negative source) ==========

def dual_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']

    # draw randomly from negative candidates and extract features
    class_to_taxa = dataset.class_to_taxa
    # Compute negatives 1
    neg1_loc = []
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        neg_candidates_id = absence_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        neg1_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    neg1_loc = torch.cat(neg1_loc, dim=0)
    neg1_feat = loc_to_feats(neg1_loc, normalize=False)
    # Compute negatives 2
    neg2_loc = []
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        neg_candidates_id = proximity_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        neg2_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    neg2_loc = torch.cat(neg2_loc, dim=0)
    neg2_feat = loc_to_feats(neg2_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, neg1_feat, neg2_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_neg1 = loc_emb_cat[batch_size:2*batch_size, :]
    loc_emb_neg2 = loc_emb_cat[2*batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_neg1 = torch.sigmoid(model.class_emb(loc_emb_neg1))
    loc_pred_neg2 = torch.sigmoid(model.class_emb(loc_emb_neg2))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_neg1 = weights * neg_log(1.0 - loc_pred_neg1[inds[:batch_size], class_id]) # assume negative
        loss_neg2 = weights * neg_log(1.0 - loc_pred_neg2[inds[:batch_size], class_id]) 
    elif neg_type == 'entropy':
        loss_neg1 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_neg1[inds[:batch_size], class_id])) # entropy
        loss_neg2 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_neg2[inds[:batch_size], class_id]))
    else:
        raise NotImplementedError
    
    # total loss
    loss = 2*loss_pos.mean() + loss_neg1.mean() + loss_neg2.mean()
    
    return loss


def dual_slds(batch, model, params, loc_to_feats, neg_type='hard'):


    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # draw randomly from negative candidates and extract features
    class_to_taxa = dataset.class_to_taxa
    # compute negatives 1: absence
    bg_class1 = []
    for loc in locs:
        # GET H3 CELL OF LOC
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # DRAW FROM NEGATIVE LIST OF H3 CELL
        neg_candidates_id = absence_slds_candidates[loc_cell]
        taxa = random.choice(neg_candidates_id)
        class_idx = class_to_taxa.index(int(taxa))
        bg_class1.append(class_idx)
    bg_class1 = torch.tensor(bg_class1, device=params['device'])
    # compute negatives 2: proximity
    bg_class2 = []
    for loc in locs:
        # GET H3 CELL OF LOC
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # DRAW FROM NEGATIVE LIST OF H3 CELL
        neg_candidates_id = proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        taxa = random.choice(neg_candidates_id)
        class_idx = class_to_taxa.index(int(taxa))
        bg_class2.append(class_idx)
    bg_class2 = torch.tensor(bg_class2, device=params['device'])

    # get location embeddings and predictions
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg1 = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class1]) # assume negative
        loss_bg2 = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class2])
    elif neg_type == 'entropy':
        loss_bg1 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class1])) # entropy
        loss_bg2 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class2]))
    else:
        raise NotImplementedError
    
    # total loss
    loss = 2*loss_pos.mean() + loss_bg1.mean() + loss_bg2.mean()
    
    return loss


def dual_full(batch, model, params, loc_to_feats, neg_type='hard'):


    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size, device=params['device'])

    # Access negative candidates from the dataset
    dataset = params['dataset']
    class_to_taxa = dataset.class_to_taxa
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # ======= Negatives =======
    rand_locs_all = []  # For all SSDL negatives
    neg_classes_all = []  # For all SLDS negatives

    for i in range(batch_size):
        taxon_id = str(class_to_taxa[class_id[i].item()])
        lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])

        # --- SSDL Negatives ---
        loc_abs_ssdl = random.choice(absence_ssdl_candidates[taxon_id])
        loc_prox_ssdl = random.choice(proximity_ssdl_candidates[taxon_id])

        rand_locs_all.append(torch.tensor(loc_abs_ssdl).view(1, -1))
        rand_locs_all.append(torch.tensor(loc_prox_ssdl).view(1, -1))

        # --- SLDS Negatives ---
        taxa_abs_slds = random.choice(absence_slds_candidates[loc_cell])
        taxa_prox_slds = random.choice(
            proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        )

        neg_classes_all.append(class_to_taxa.index(int(taxa_abs_slds)))
        neg_classes_all.append(class_to_taxa.index(int(taxa_prox_slds)))
    
    # === Convert SSDL negative coords to features ===
    rand_locs_all = torch.cat(rand_locs_all, dim=0).to(params['device'])     # Shape: [2B, 2]
    rand_feat_all = loc_to_feats(rand_locs_all, normalize=False)   

    # get location embeddings and predictions
    loc_cat = torch.cat((loc_feat, rand_feat_all), dim=0)
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size]
    loc_emb_rand_all = loc_emb_cat[batch_size:]

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))  # [B, num_classes]
    loc_pred_rand_all = torch.sigmoid(model.class_emb(loc_emb_rand_all))

    neg_classes_all = torch.tensor(neg_classes_all, device=params['device']) 

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # === Loss ===
    loss_pos = weights * neg_log(loc_pred[inds, class_id])

    # Negative SSDL: absence + proximity
    loss_neg_ssdl_abs = weights * neg_log(1.0 - loc_pred_rand_all[0::2, class_id])  # abs SSDL
    loss_neg_ssdl_prox = weights * neg_log(1.0 - loc_pred_rand_all[1::2, class_id])  # prox SSDL

    # Negative SLDS: absence + proximity
    loss_neg_slds_abs = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[0::2]])
    loss_neg_slds_prox = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[1::2]])

    # Total loss
    loss = (
        4*loss_pos.mean() 
        + loss_neg_ssdl_abs.mean() + loss_neg_ssdl_prox.mean()
        + loss_neg_slds_abs.mean() + loss_neg_slds_prox.mean()
    )
    
    return loss


# ========== DUAL WITH SPECIES PREFERENCE FOR ABSENCE ==========


def dual_smart_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']

    # draw randomly from negative candidates and extract features
    class_to_taxa = dataset.class_to_taxa
    # preference for absence (weights)
    p_species = params['dataset'].p_abs
    # Compute negatives 1
    neg1_loc = []
    weights_abs = []
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        #
        weight_taxa = p_species.get(taxon_id, 1)
        weights_abs.append(weight_taxa)
        #
        neg_candidates_id = absence_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        neg1_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    neg1_loc = torch.cat(neg1_loc, dim=0)
    neg1_feat = loc_to_feats(neg1_loc, normalize=False)
    # Compute negatives 2
    neg2_loc = []
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        neg_candidates_id = proximity_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        neg2_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    neg2_loc = torch.cat(neg2_loc, dim=0)
    neg2_feat = loc_to_feats(neg2_loc, normalize=False)

    weights_abs = torch.tensor(weights_abs, dtype=torch.float32, device=params['device'])
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, neg1_feat, neg2_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_neg1 = loc_emb_cat[batch_size:2*batch_size, :]
    loc_emb_neg2 = loc_emb_cat[2*batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_neg1 = torch.sigmoid(model.class_emb(loc_emb_neg1))
    loc_pred_neg2 = torch.sigmoid(model.class_emb(loc_emb_neg2))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_neg1 = weights * neg_log(1.0 - loc_pred_neg1[inds[:batch_size], class_id]) * weights_abs 
        loss_neg2 = weights * neg_log(1.0 - loc_pred_neg2[inds[:batch_size], class_id]) * (1-weights_abs)
    elif neg_type == 'entropy':
        loss_neg1 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_neg1[inds[:batch_size], class_id])) * weights_abs
        loss_neg2 = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_neg2[inds[:batch_size], class_id])) * (1-weights_abs)
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_neg1.mean() + loss_neg2.mean()
    
    return loss


def dual_smart_slds(batch, model, params, loc_to_feats, neg_type='hard'):


    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size, device=params['device'])

    # Access negative candidates from the dataset
    dataset = params['dataset']
    class_to_taxa = dataset.class_to_taxa
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # preference for absence (weights)
    p_species = params['dataset'].p_abs

    # ======= Negatives =======
    rand_locs_all = []  # For all SSDL negatives
    neg_classes_all = []  # For all SLDS negatives
    weights_abs = []

    for i in range(batch_size):
        taxon_id = str(class_to_taxa[class_id[i].item()])
        lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])

        #
        weight_taxa = p_species.get(taxon_id, 1)
        weights_abs.append(weight_taxa)
        #

        # --- SLDS Negatives ---
        taxa_abs_slds = random.choice(absence_slds_candidates[loc_cell])
        taxa_prox_slds = random.choice(
            proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        )

        neg_classes_all.append(class_to_taxa.index(int(taxa_abs_slds)))
        neg_classes_all.append(class_to_taxa.index(int(taxa_prox_slds)))
    weights_abs = torch.tensor(weights_abs, dtype=torch.float32, device=params['device'])
    

    # get location embeddings and predictions
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    neg_classes_all = torch.tensor(neg_classes_all, device=params['device']) 

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # === Loss ===
    loss_pos = weights * neg_log(loc_pred[inds, class_id])

    # Negative SLDS: absence + proximity
    loss_neg_slds_abs = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[0::2]]) * weights_abs 
    loss_neg_slds_prox = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[1::2]]) * (1-weights_abs) 
    
    # total loss
    loss = loss_pos.mean() + loss_neg_slds_abs.mean() + loss_neg_slds_prox.mean()
    
    return loss


def dual_smart_full(batch, model, params, loc_to_feats, neg_type='hard'):


    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size, device=params['device'])

    # Access negative candidates from the dataset
    dataset = params['dataset']
    class_to_taxa = dataset.class_to_taxa
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # preference for absence (weights)
    p_species = params['dataset'].p_abs

    # ======= Negatives =======
    rand_locs_all = []  # For all SSDL negatives
    neg_classes_all = []  # For all SLDS negatives
    weights_abs = []

    for i in range(batch_size):
        taxon_id = str(class_to_taxa[class_id[i].item()])
        lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])

        #
        weight_taxa = p_species.get(taxon_id, 1)
        weights_abs.append(weight_taxa)
        #

        # --- SSDL Negatives ---
        loc_abs_ssdl = random.choice(absence_ssdl_candidates[taxon_id])
        loc_prox_ssdl = random.choice(proximity_ssdl_candidates[taxon_id])

        rand_locs_all.append(torch.tensor(loc_abs_ssdl).view(1, -1))
        rand_locs_all.append(torch.tensor(loc_prox_ssdl).view(1, -1))

        # --- SLDS Negatives ---
        taxa_abs_slds = random.choice(absence_slds_candidates[loc_cell])
        taxa_prox_slds = random.choice(
            proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        )

        neg_classes_all.append(class_to_taxa.index(int(taxa_abs_slds)))
        neg_classes_all.append(class_to_taxa.index(int(taxa_prox_slds)))
    weights_abs = torch.tensor(weights_abs, dtype=torch.float32, device=params['device'])
    
    # === Convert SSDL negative coords to features ===
    rand_locs_all = torch.cat(rand_locs_all, dim=0).to(params['device'])     # Shape: [2B, 2]
    rand_feat_all = loc_to_feats(rand_locs_all, normalize=False)   

    # get location embeddings and predictions
    loc_cat = torch.cat((loc_feat, rand_feat_all), dim=0)
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size]
    loc_emb_rand_all = loc_emb_cat[batch_size:]

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))  # [B, num_classes]
    loc_pred_rand_all = torch.sigmoid(model.class_emb(loc_emb_rand_all))

    neg_classes_all = torch.tensor(neg_classes_all, device=params['device']) 

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # === Loss ===
    loss_pos = weights * neg_log(loc_pred[inds, class_id])

    # Negative SSDL: absence + proximity
    loss_neg_ssdl_abs = weights * neg_log(1.0 - loc_pred_rand_all[0::2, class_id]) * weights_abs  # abs SSDL
    loss_neg_ssdl_prox = weights * neg_log(1.0 - loc_pred_rand_all[1::2, class_id]) * (1-weights_abs)  # prox SSDL

    # Negative SLDS: absence + proximity
    loss_neg_slds_abs = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[0::2]]) * weights_abs 
    loss_neg_slds_prox = weights * neg_log(1.0 - loc_pred[inds, neg_classes_all[1::2]]) * (1-weights_abs) 

    # Total loss
    loss = (
        2*loss_pos.mean() 
        + loss_neg_ssdl_abs.mean() + loss_neg_ssdl_prox.mean()
        + loss_neg_slds_abs.mean() + loss_neg_slds_prox.mean()
    )
    
    return loss



# ========== HYBRID WITH SPECIES PREFERENCE FOR ABSENCE ==========

def hybrid_smart_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']

    # draw randomly from negative candidates and extract features
    rand_loc = []
    class_to_taxa = dataset.class_to_taxa
    # probability of drawing from absence
    p_species = params['dataset'].p_abs
    # Compute negatives
    for class_id_val in class_id:
        taxon_id = str(class_to_taxa[class_id_val.item()])
        p_taxa = p_species.get(taxon_id, 1)
        if np.random.rand() < p_taxa: # select absence negative
            neg_candidates_id = absence_ssdl_candidates[taxon_id]
        else:   # select proximity negative
            neg_candidates_id = proximity_ssdl_candidates[taxon_id]
        loc = random.choice(neg_candidates_id)
        rand_loc.append(torch.tensor(loc).view(1, -1))  # Ensures shape [1, 2]
    rand_loc = torch.cat(rand_loc, dim=0)
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def hybrid_smart_slds(batch, model, params, loc_to_feats, neg_type='hard'):


    inds = torch.arange(params['batch_size'])

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']

    # draw randomly from negative candidates and extract features
    bg_class = []
    class_to_taxa = dataset.class_to_taxa
    # probability of drawing from absence
    p_species = params['dataset'].p_abs
    for i in range(batch_size):
        # extract values
        class_id_val = class_id[i]
        loc = locs[i]
        taxon_id = str(class_to_taxa[class_id_val.item()])
        p_taxa = p_species.get(taxon_id, 1)
        # GET H3 CELL OF LOC
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # DRAW FROM NEGATIVE LIST OF H3 CELL
        if np.random.rand() < p_taxa: # 0.5 chance of absence negative
            neg_candidates_id = absence_slds_candidates[loc_cell]
        else:   # 0.5 chance of proximity negative
            neg_candidates_id = proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        taxa = random.choice(neg_candidates_id)
        class_idx = class_to_taxa.index(int(taxa))
        bg_class.append(class_idx)
    bg_class = torch.tensor(bg_class, device=params['device'])

    # get location embeddings and predictions
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = weights * neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss



def hybrid_smart_full(batch, model, params, loc_to_feats, neg_type='hard'):

    loc_feat, locs, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size, device=params['device'])

    # Access negative candidates from the dataset
    dataset = params['dataset']
    absence_ssdl_candidates = dataset.candidates['absence_ssdl']
    proximity_ssdl_candidates = dataset.candidates['proximity_ssdl']
    absence_slds_candidates = dataset.candidates['absence_slds']
    proximity_slds_candidates = dataset.candidates['proximity_slds']


    # ======= Negatives =======
    class_to_taxa = dataset.class_to_taxa
    # Probability of drawing from absence
    p_species = params['dataset'].p_abs
    rand_loc = []
    neg_classes = []
    for i in range(batch_size):
        # extract values
        class_id_val = class_id[i]
        taxon_id = str(class_to_taxa[class_id_val.item()])
        p_taxa = p_species.get(taxon_id, 1)
        loc = locs[i]
        lon, lat = loc[0].item() * 180.0, loc[1].item() * 90.0
        loc_cell = h3.geo_to_h3(lat, lon, params['h3_resolution'])
        # select absence or proximity
        if np.random.rand() < p_taxa:
            neg_candidates_id_ssdl = absence_ssdl_candidates[taxon_id]
            neg_candidates_id_slds = absence_slds_candidates[loc_cell]
        else:
            neg_candidates_id_ssdl = proximity_ssdl_candidates[taxon_id]
            neg_candidates_id_slds = proximity_slds_candidates[loc_cell] if loc_cell in proximity_slds_candidates else absence_slds_candidates[loc_cell]
        # Location features for SSDL:
        loc = random.choice(neg_candidates_id_ssdl)
        rand_loc.append(torch.tensor(loc).view(1, -1))
         # Append background class for SLDS
        taxa = random.choice(neg_candidates_id_slds)
        neg_classes.append(class_to_taxa.index(int(taxa)))

    # Extract features background location
    rand_loc = torch.cat(rand_loc, dim=0).to(params['device'])
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # Background classes to tensor
    neg_classes = torch.tensor(neg_classes, device=params['device'])
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # Location weights
    weights = torch.ones(batch_size, device=params['device'])
    if params['weighted']:
        # Compute weights based on H3 cell counts
        h3_res2_presences = params['dataset'].h3_count
        for i in range(batch_size):
            lon, lat = locs[i, 0].item() * 180.0, locs[i, 1].item() * 90.0
            h3_cell = h3.geo_to_h3(lat, lon, resolution=params['h3_resolution'])
            count = h3_res2_presences.get(h3_cell, 1)  # Default to 1 if not found
            weights[i] = min(1.0, 1 / np.log(count + 1)) 
        weights = weights * batch_size / weights.sum()  # Normalize weights
    
    # data loss
    loss_pos = weights * neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_neg_ssdl = weights * neg_log(1.0 - loc_pred_rand[inds, class_id])
        loss_neg_slds = weights * neg_log(1.0 - loc_pred[inds, neg_classes])
    elif neg_type == 'entropy':
        loss_neg_ssdl = weights * (-1 * bernoulli_entropy(1.0 - loc_pred_rand[inds, class_id]))
        loss_neg_slds = weights * (-1 * bernoulli_entropy(1.0 - loc_pred[inds, neg_classes]))
    else:
        raise NotImplementedError

    # Total loss
    loss = 2*loss_pos.mean() + loss_neg_ssdl.mean() + loss_neg_slds.mean()
    
    return loss


