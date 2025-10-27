import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import utils
import models
import datasets


def load_prediction(eval_params, paths):
    """Load model predictions for the given taxa using the old viz_map logic."""
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    model.eval()

    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env()
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    if eval_params['taxa_id'] in train_params['params']['class_to_taxa']:
        class_of_interest = train_params['params']['class_to_taxa'].index(eval_params['taxa_id'])
    else:
        raise ValueError(f"Taxa {eval_params['taxa_id']} not in model")

    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    locs = utils.coord_grid(mask.shape)
    if not eval_params['disable_ocean_mask']:
        locs = locs[mask_inds, :]
    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(eval_params['device'])

    with torch.no_grad():
        preds = model(locs_enc, return_feats=False, class_of_interest=class_of_interest).cpu().numpy()

    if eval_params['threshold'] > 0:
        preds[preds < eval_params['threshold']] = 0.0
        preds[preds >= eval_params['threshold']] = 1.0

    if not eval_params['disable_ocean_mask']:
        op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan
        op_im[mask_inds] = preds
    else:
        op_im = preds

    return op_im.reshape(mask.shape)


def load_iucn(eval_params, paths):
    """Load IUCN ground truth map."""
    with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
        data = json.load(f)

    taxa = [int(tt) for tt in data['taxa_presence'].keys()]
    if eval_params['taxa_id'] not in taxa:
        raise ValueError(f"Taxa {eval_params['taxa_id']} not in IUCN dataset")

    locs = np.array(data['locs'], dtype=np.float32)

    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    grid_locs = utils.coord_grid(mask.shape)
    tree = cKDTree(grid_locs)

    gt_grid = np.zeros(grid_locs.shape[0], dtype=np.float32)
    presence_indices = data['taxa_presence'][str(eval_params['taxa_id'])]
    presence_locations = locs[presence_indices]

    if len(presence_locations) > 0:
        distances, multi_indices = tree.query(presence_locations, k=min(4, grid_locs.shape[0]))
        distance_threshold = 0.5
        for dists, idxs in zip(distances, multi_indices):
            if np.ndim(dists) == 0:
                if dists <= distance_threshold:
                    gt_grid[idxs] = 1.0
            else:
                valid_indices = idxs[dists <= distance_threshold]
                gt_grid[valid_indices] = 1.0

    combined_im = np.zeros(mask.shape[0] * mask.shape[1])
    if not eval_params['disable_ocean_mask']:
        combined_im[mask_inds] = 1  # land
        presence_mask_indices = mask_inds[gt_grid[mask_inds] == 1]
        combined_im[presence_mask_indices] = 2  # presence
    else:
        combined_im[:] = 1
        combined_im[gt_grid == 1] = 2

    return combined_im.reshape(mask.shape)


def main(eval_params):
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    pred_map = load_prediction(eval_params, paths)
    iucn_map = load_iucn(eval_params, paths)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Background: ocean light grey, land white
    background = np.zeros_like(iucn_map)
    background[iucn_map == 0] = 0  # ocean
    background[iucn_map >= 1] = 1  # land
    bg_cmap = ListedColormap(['lightgrey', 'white'])
    ax.imshow(background, extent=[-180, 180, -90, 90], cmap=bg_cmap, vmin=0, vmax=1)
    # Prediction heatmap
    im_pred = ax.imshow(pred_map, extent=[-180, 180, -90, 90],
                    cmap=plt.cm.magma, vmin=0, vmax=1, alpha=0.5)
    """
    # Add inset colorbar at bottom
    axins = inset_axes(ax,
                   width="30%",  # 30% of parent axes
                   height="3%",  # small bar
                   loc='lower center',  # position inside axes
                   borderpad=4)  # padding from edge
    cbar = plt.colorbar(im_pred, cax=axins, orientation="horizontal")
    cbar.set_label("Predicted suitability", fontsize=12)
    cbar.ax.tick_params(labelsize=8)
    """

    # IUCN contours (presence = 2)
    presence_mask = (iucn_map == 2).astype(float)
    # We need to generate coordinate grids for lon/lat matching the mask
    nlat, nlon = presence_mask.shape
    lons = np.linspace(-180, 180, nlon)
    lats = np.linspace(90, -90, nlat)
    cs = ax.contour(lons, lats, presence_mask,
                levels=[0.5],
                colors='red', linewidths=1)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title(f"Prediction vs IUCN for taxa {eval_params['taxa_id']}")

    save_loc = os.path.join(eval_params['op_path'], f'overlay_{eval_params["taxa_id"]}.png')
    print(f"Saving overlay to {save_loc}")
    plt.savefig(save_loc, bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--taxa_id', type=int, required=True)
    parser.add_argument('--op_path', type=str, default='./images/')
    parser.add_argument('--threshold', type=float, default=-1)
    parser.add_argument('--high_res', action='store_true')
    parser.add_argument('--disable_ocean_mask', action='store_true')
    parser.add_argument('--set_max_cmap_to_1', action='store_true')
    parser.add_argument('--device', type=str, default=device)
    eval_params = vars(parser.parse_args())

    if not os.path.isdir(eval_params['op_path']):
        os.makedirs(eval_params['op_path'])

    main(eval_params)
