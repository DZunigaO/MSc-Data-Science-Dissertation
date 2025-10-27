"""
Demo that takes an iNaturalist taxa ID as input and generates a presence-absence map
for the specified taxa using the IUCN dataset, saving the output as an image.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap

import utils

def main(eval_params):
    # Load paths
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    # Load IUCN data
    with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
        data = json.load(f)

    # Extract taxa and locations
    taxa = [int(tt) for tt in data['taxa_presence'].keys()]
    locs = np.array(data['locs'], dtype=np.float32)

    # Check if taxa_id is valid
    if eval_params['taxa_id'] not in taxa:
        print(f'Error: Taxa ID {eval_params["taxa_id"]} not found in IUCN dataset.')
        return False
    print(f'Loading taxa: {eval_params["taxa_id"]}')

    # Load ocean mask
    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    # Generate grid coordinates (same as in viz_map.py)
    grid_locs = utils.coord_grid(mask.shape)

    # Map IUCN locs to the closest grid points
    tree = cKDTree(grid_locs)
    _, indices = tree.query(locs, k=1) 

    # Generate ground truth presence-absence data for the grid
    gt_grid = np.zeros(grid_locs.shape[0], dtype=np.float32)
    presence_indices = data['taxa_presence'][str(eval_params['taxa_id'])]



    # Set presence (1) for grid points corresponding to presence locations
    presence_locations = locs[presence_indices]
    if len(presence_locations) > 0:
        # Query multiple nearest grid points for each presence location
        distances, multi_indices = tree.query(presence_locations, k=min(4, grid_locs.shape[0]))
        
        # Set presence for grid points within a reasonable distance
        # Adjust the distance threshold based on your grid resolution
        distance_threshold = 0.5  # degrees - adjust as needed
        
        for i, (dists, idxs) in enumerate(zip(distances, multi_indices)):
            if dists.ndim == 0:  # Single nearest neighbor
                if dists <= distance_threshold:
                    gt_grid[idxs] = 1.0
            else:  # Multiple nearest neighbors
                valid_indices = idxs[dists <= distance_threshold]
                gt_grid[valid_indices] = 1.0

    # Create a combined visualization array
    # 0 = ocean (white), 1 = land (grey), 2 = presence (yellow)
    combined_im = np.zeros((mask.shape[0] * mask.shape[1]))
    
    if not eval_params['disable_ocean_mask']:
        # Set land areas to 1 (grey)
        combined_im[mask_inds] = 1
        # Set presence areas to 2 (yellow)
        presence_mask_indices = mask_inds[gt_grid[mask_inds] == 1]
        combined_im[presence_mask_indices] = 2
    else:
        # Without ocean mask, everything is either land or presence
        combined_im[:] = 1  # Default to land
        combined_im[gt_grid == 1] = 2  # Presence areas
    
    combined_im = combined_im.reshape(mask.shape)

    # Create custom colormap: white (ocean), grey (land), yellow (presence)
    colors = ['white', 'grey', 'purple']
    cmap = ListedColormap(colors)

    # Set up figure for visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the combined image
    im = ax.imshow(combined_im, extent=[-180, 180, -90, 90], cmap=cmap, vmin=0, vmax=2)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove the frame/spines if desired
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save image
    save_loc = os.path.join(eval_params['op_path'], f'map_iucn_{eval_params["taxa_id"]}.png')
    print(f'Saving image to {save_loc}')
    plt.savefig(save_loc, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    return True

if __name__ == '__main__':
    info_str = '\nDemo that takes an iNaturalist taxa ID as input and ' + \
               'generates a presence-absence map for the specified taxa ' + \
               'using the IUCN dataset, saving the output as an image.\n\n' + \
               'Note: This script visualizes ground truth presence-absence data from IUCN.'

    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--taxa_id', type=int, default=14, help='iNaturalist taxon ID.')
    parser.add_argument('--op_path', type=str, default='./images/', help='Location where the output image will be saved.')
    parser.add_argument('--high_res', action='store_true', help='Generate higher resolution output.')
    parser.add_argument('--disable_ocean_mask', action='store_true', help='Do not use an ocean mask.')
    parser.add_argument('--set_max_cmap_to_1', action='store_true', help='Set maximum intensity to 1 for consistent output.')
    eval_params = vars(parser.parse_args())

    if not os.path.isdir(eval_params['op_path']):
        os.makedirs(eval_params['op_path'])

    main(eval_params)