import torch
import numpy as np
import pandas as pd
import math
import datetime
import h3
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point, shape
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import os
import json

class CoordEncoder:

    def __init__(self, input_enc, raster=None):
        self.input_enc = input_enc
        self.raster = raster

    def encode(self, locs, normalize=True):
        # assumes lon, lat in range [-180, 180] and [-90, 90]
        if normalize:
            locs = normalize_coords(locs)
        if self.input_enc == 'sin_cos': # sinusoidal encoding
            loc_feats = encode_loc(locs)
        elif self.input_enc == 'env': # bioclim variables
            loc_feats = bilinear_interpolate(locs, self.raster)
        elif self.input_enc == 'sin_cos_env': # sinusoidal encoding & bioclim variables
            loc_feats = encode_loc(locs)
            context_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((loc_feats, context_feats), 1)
        else:
            raise NotImplementedError('Unknown input encoding.')
        return loc_feats

def normalize_coords(locs):
    # locs is in lon {-180, 180}, lat {90, -90}
    # output is in the range [-1, 1]

    locs[:,0] /= 180.0
    locs[:,1] /= 90.0

    return locs

def encode_loc(loc_ip, concat_dim=1):
    # assumes inputs location are in range -1 to 1
    # location is lon, lat
    feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), concat_dim)
    return feats

def bilinear_interpolate(loc_ip, data, remove_nans_raster=True):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    assert data is not None

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1] # this is because latitude goes from +90 on top to bottom while
                            # longitude goes from -90 to 90 left to right

    assert not torch.any(torch.isnan(loc))

    if remove_nans_raster:
        data[torch.isnan(data)] = 0.0 # replace with mean value (0 is mean post-normalization)

    # cast locations into pixel space
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val

def rand_samples(batch_size, device, rand_type='uniform'):
    # randomly sample background locations (returns lat & lon in the range [-1,1])
    torch.manual_seed(626)

    if rand_type == 'spherical':
        rand_loc = torch.rand(batch_size, 2).to(device)
        theta1 = 2.0*math.pi*rand_loc[:, 0]
        theta2 = torch.acos(2.0*rand_loc[:, 1] - 1.0)
        lat = 1.0 - 2.0*theta2/math.pi
        lon = (theta1/math.pi) - 1.0
        rand_loc = torch.cat((lon.unsqueeze(1), lat.unsqueeze(1)), 1)

    elif rand_type == 'uniform':
        rand_loc = torch.rand(batch_size, 2).to(device)*2.0 - 1.0

    return rand_loc

def get_time_stamp():
    cur_time = str(datetime.datetime.now())
    date, time = cur_time.split(' ')
    h, m, s = time.split(':')
    s = s.split('.')[0]
    time_stamp = '{}-{}-{}-{}'.format(date, h, m, s)
    return time_stamp

def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    # generate a grid of locations spaced evenly in coordinate space

    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(np.linspace(-180, 180, feats.shape[1]), np.linspace(90, -90, feats.shape[0]))
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        # return feats for all locations
        # this will be an N x 2 array
        return feats.reshape(feats.shape[0]*feats.shape[1], 2)
    else:
        # only select a subset of locations
        ind_y, ind_x = np.where(split_ids==split_of_interest)

        # these will be N_subset x 2 in size
        return feats[ind_y, ind_x, :]

def create_spatial_split(raster, mask, train_amt=1.0, cell_size=25):
    np.random.seed(626)
    # generates a checkerboard style train test split
    # 0 is invalid, 1 is train, and 2 is test
    # c_size is units of pixels
    split_ids = np.ones((raster.shape[0], raster.shape[1]))
    start = cell_size
    for ii in np.arange(0, split_ids.shape[0], cell_size):
        if start == 0:
            start = cell_size
        else:
            start = 0
        for jj in np.arange(start, split_ids.shape[1], cell_size*2):
            split_ids[ii:ii+cell_size, jj:jj+cell_size] = 2
    split_ids = split_ids*mask
    if train_amt < 1.0:
        # take a subset of the data
        tr_y, tr_x = np.where(split_ids==1)
        inds = np.random.choice(len(tr_y), int(len(tr_y)*(1.0-train_amt)), replace=False)
        split_ids[tr_y[inds], tr_x[inds]] = 0
    return split_ids

def average_precision_score_faster(y_true, y_scores):
    # drop in replacement for sklearn's average_precision_score
    # comparable up to floating point differences
    num_positives = y_true.sum()
    inds = np.argsort(y_scores)[::-1]
    y_true_s = y_true[inds]

    false_pos_c = np.cumsum(1.0 - y_true_s)
    true_pos_c = np.cumsum(y_true_s)
    recall = true_pos_c / num_positives
    false_neg = np.maximum(true_pos_c + false_pos_c, np.finfo(np.float32).eps)
    precision = true_pos_c / false_neg

    recall_e = np.hstack((0, recall, 1))
    recall_e = (recall_e[1:] - recall_e[:-1])[:-1]
    map_score = (recall_e*precision).sum()
    return map_score


def binary_metrics_old(y_true, y_scores):
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
    best_threshold = np.min(y_scores)
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_p = precision
            best_r = recall
            best_threshold = threshold
    return best_p, best_r, best_threshold

from sklearn.metrics import precision_recall_curve
def binary_metrics(y_true, y_scores, max_thresholds=50):
    # Get unique sorted scores to reduce the number of thresholds
    unique_scores = np.unique(y_scores)
    if len(unique_scores) > max_thresholds:
        thresholds = np.linspace(unique_scores.min(), unique_scores.max(), max_thresholds)
    else:
        thresholds = unique_scores
    # Compute precision, recall, and thresholds using scikit-learn
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # Compute F1 scores for all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Find the index of the maximum F1 score
    valid_idx = ~np.isnan(f1_scores)
    if not np.any(valid_idx):
        return np.nan, np.nan, np.nan  # No valid F1 scores (e.g., no positive predictions)
    best_idx = np.argmax(f1_scores[valid_idx])
    best_p = precision[valid_idx][best_idx]
    best_r = recall[valid_idx][best_idx]
    # Get the corresponding threshold
    # precision_recall_curve returns thresholds for all but the last precision/recall pair
    best_threshold = thresholds[valid_idx[:-1]][best_idx] if len(thresholds) > 0 else np.min(y_scores)
    return best_p, best_r, best_threshold

def confusion_values(y_true, y_scores, h3_cells):
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_p = precision
            best_r = recall
            best_threshold = threshold
    # Final predictions at best threshold
    y_pred_final = (y_scores >= best_threshold).astype(int)
    # Collect per-cell results
    cell_results = []
    for i in range(len(y_true)):
        cell_results.append({
            'h3': h3_cells[i],
            'y_true': int(y_true[i]),
            'y_pred': int(y_pred_final[i]),
            'score': float(y_scores[i])
        })
    return best_p, best_r, cell_results


def geo_convex_hull(locs, eps=600, min_samples=5): # ADDED FOR CONVEX HULL
    # Coordinates clusters
    kms_per_radian = 6371.0088
    epsilon = eps / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(locs))
    cluster_labels = db.labels_
    # Convex Hull for each cluster
    hulls = []
    for cluster_id in cluster_labels:
        if cluster_id != -1: # -1 are noise points not included in any cluster
            points = [pt for pt, label in zip(locs, cluster_labels) if label == cluster_id]
            if len(points) >= 3: # 3 is the min for a shape, we could try with larger number for more comp
                hull = MultiPoint([(lon, lat) for lat, lon in points]).convex_hull
                coords = list(hull.exterior.coords)
                hulls.append([[[lon, lat] for lon, lat in coords]])
    return hulls

def rand_samples_informed(mask, device): # ADDED FOR DRAWING RANDOM COORDS FROM SUBSET MASK
    # considering prior mask, randomly sample background locations (returns lat & lon in the range [-1,1])
    torch.manual_seed(627)
    polygon = shape(mask)
    while True:
        rand_loc = torch.rand(2).to(device)
        theta1 = 2.0*math.pi*rand_loc[0]
        theta2 = torch.acos(2.0*rand_loc[1] - 1.0)
        lat = 1.0 - 2.0*theta2/math.pi
        lon = (theta1/math.pi) - 1.0
        # Convert to full degree coordinates
        lat_deg = lat * 90.0
        lon_deg = lon * 180.0
        point = Point(lon_deg, lat_deg)
        if polygon.contains(point):
           return torch.tensor([lon, lat])
        
def batched_forward(model, loc_feat, batch_size=1024):
    loc_emb_list = []
    for i in range(0, loc_feat.shape[0], batch_size):
        loc_emb_list.append(model(loc_feat[i:i+batch_size], return_feats=True).cpu())
    return torch.cat(loc_emb_list, dim=0)


def generate_masks(cells_file, proximity_k = 2):
    world = Polygon([[-180, -90], [180, -90], [180, 90], [-180, 90]])
    presence_masks, absence_masks, proximity_masks = {}, {}, {}
    def fix_dateline_crossing(coords):
        """
        Adjusts longitudes to avoid issues with meridian crossing
        """
        lons = [lon for lon, lat in coords]
        if max(lons) - min(lons) > 180:
            # Likely dateline crossing: convert all longitudes to [0, 360]
            fixed_coords = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in coords]
        else:
            fixed_coords = coords
        return fixed_coords
    for species_id, cells_list in cells_file.items():
        # --- Presence Mask ---
        cell_polys = []
        for cell in cells_list:
            boundary = h3.h3_to_geo_boundary(cell, geo_json=True)  # list of [lat, lon]
            fixed_boundary = fix_dateline_crossing(boundary) 
            poly = Polygon(fixed_boundary)
            if not poly.is_valid:
                poly = make_valid(poly)  # Fix invalid geometry
            cell_polys.append(poly)    
        if not cell_polys:
            poly = world
            presence_masks[species_id] = {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
            absence_masks[species_id] = {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
            proximity_masks[species_id] = {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
            continue
        presence_union = unary_union(cell_polys)
        presence_masks[species_id] = presence_union.__geo_interface__
        # --- Absence Mask ---
        absence = world.difference(presence_union)
        absence_masks[species_id] = absence.__geo_interface__
        # --- Proximity Mask ---
        neighbors = set()
        for cell in cells_list:
            ring = h3.k_ring(cell, proximity_k)
            ring.discard(cell)
            neighbors.update(ring)
        proximity_cells = neighbors - set(cells_list)
        cell_polys = []
        for cell in proximity_cells:
            boundary = h3.h3_to_geo_boundary(cell, geo_json=True)  # list of [lat, lon]
            fixed_boundary = fix_dateline_crossing(boundary) 
            poly = Polygon(fixed_boundary)
            if not poly.is_valid:
                poly = make_valid(poly)  # Fix invalid geometry
            cell_polys.append(poly)        
        proximity_union = unary_union(cell_polys).difference(presence_union.buffer(0))
        proximity_masks[species_id] = proximity_union.__geo_interface__
    # Save all masks
    os.makedirs("data/train", exist_ok=True)
    with open("data/train/presence_masks_cells.json", "w") as f:
        json.dump(presence_masks, f)
    with open("data/train/absence_masks_cells.json", "w") as f:
        json.dump(absence_masks, f)
    with open("data/train/proximity_masks_cells.json", "w") as f:
        json.dump(proximity_masks, f)
    return presence_masks, absence_masks, proximity_masks

def sample_unit_sphere_coord():
    torch.manual_seed(628)
    rand_loc = torch.rand(2)
    theta1 = 2.0 * math.pi * rand_loc[0]
    theta2 = torch.acos(2.0 * rand_loc[1] - 1.0)
    lat = 1.0 - 2.0 * theta2 / math.pi
    lon = (theta1 / math.pi) - 1.0
    return [lon.item(), lat.item()]

def sample_points_in_mask(mask, num_points=1000, max_attempts=100000):
    candidates = {}
    for taxon_id, mask in mask.items():
        polygon = shape(mask)
        points = []
        attempts = 0
        while len(points) < num_points and attempts < max_attempts:
            lonlat = sample_unit_sphere_coord()
            point = Point(lonlat[0] * 180.0, lonlat[1] * 90.0)  # convert to deg
            if polygon.contains(point):
                points.append(lonlat)
            attempts += 1
        candidates[taxon_id] = points
    return candidates


def absence_matrix(class_to_taxa, cell_list, presence_dict):
    import gen_negative_candidates
    from sklearn.feature_extraction import DictVectorizer
    # presence cells -> species
    cell_to_species  = gen_negative_candidates.invert_species_to_cells(presence_dict)
    num_locs = len(cell_list)
    num_classes = len(class_to_taxa)
    # absence matrix
    absence_mat = torch.ones((num_locs, num_classes)) 
    for i, cell in enumerate(cell_list):
        if cell in cell_to_species:
            for sp_str in cell_to_species[cell]:
                try:
                    class_idx = class_to_taxa.index(int(sp_str))
                    absence_mat[i, class_idx] = 0  # 0 if presence
                except ValueError:
                    continue  # species not in current class list
    return absence_mat


def get_p_absence(n_cells, kl, model_path="data/train/region_model.json"):
    with open(model_path, "r") as f:
        model = json.load(f)

    mean = np.array(model["scaler_inputs_mean"])
    scale = np.array(model["scaler_inputs_scale"])
    centroids = np.array(model["centroids"])
    region_stats = model["region_stats"]

    inputs = np.column_stack([kl, n_cells])
    inputs_scaled = (inputs - mean) / scale

    preds = []
    for point in inputs_scaled:
        distances = np.linalg.norm(centroids[:, :2] - point[:2], axis=1)
        closest = int(np.argmin(distances))
        preds.append(region_stats[str(closest)]["mean_p_abs"])
    return np.array(preds)



