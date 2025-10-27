import numpy as np
import h3
import os
import json
from shapely.geometry import Polygon,  mapping, Point
import random

import datasets

np.random.seed(42)
random.seed(42)

def get_h3_cells(class_to_taxa, class_ids, locs, cell_resolution=1, max_per_species=1000):
    """
    Function stores the h3 cells (of selected resolution) in which each species have been observed
    """
    cells_per_species = {}
    for i, taxon_id in enumerate(class_to_taxa):
        idx = np.where(class_ids == i)[0]
        species_locs = locs[idx]
        if len(species_locs) > max_per_species:
            np.random.shuffle(species_locs)
            species_locs = species_locs[:max_per_species]
        h3_cells = list(set([h3.geo_to_h3(lat, lon, cell_resolution) for lon, lat in species_locs]))
        cells_per_species[taxon_id] = h3_cells
    return cells_per_species

def gen_negative_cells(presence_cells_dict, h3_resolution=1, proximity_k=2):
    """
    From presence cells, derive absence and proximity cells for each species
    """
    absence_cells = {}
    proximity_cells = {}
    # Get list of worldwide H3 cells
    full_index = set(full_h3_list(resolution=h3_resolution))
    for species_id, presence_list in presence_cells_dict.items():
        presence_set = set(presence_list)
        # Proximity = cells adjacent to presence
        neighbors = set()
        for cell in presence_list:
            neighbors.update(h3.k_ring(cell, proximity_k))
        proximity_set = neighbors - presence_set
        # Absence = all used cells not in presence
        absence_set = full_index - presence_set
        # Add to dictionary
        absence_cells[species_id] = list(absence_set)
        proximity_cells[species_id] = list(proximity_set)
    return absence_cells, proximity_cells


def invert_species_to_cells(cells_dict):
    """
    Converts {species_id: [h3_cells]} to {h3_cell: [species_id, ...]}
    """
    cell_to_species = {}
    for species_id, cell_list in cells_dict.items():
        for cell in cell_list:
            if cell not in cell_to_species:
                cell_to_species[cell] = []
            cell_to_species[cell].append(species_id)
    return cell_to_species


def full_h3_list(resolution=1):
    # Define a world-covering polygon
    res0_cells = h3.get_res0_indexes()
    all_cells = []
    for cell in res0_cells:
        if resolution == 0:
            all_cells.append(cell)
        else:
            children = h3.h3_to_children(cell, resolution)
            all_cells.extend(children)
    return list(set(all_cells))


def sample_point_in_polygon(poly_coords):
    """
    Draw random coordinate inside cell boundaries
    """
    polygon = Polygon(poly_coords)
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(100):  # Try up to 100 times probably with 5-10 should be enough
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return [p.x, p.y]
    # fallback to centroid if all else fails
    centroid = polygon.centroid
    return [centroid.x, centroid.y]


def sample_points_from_h3_cells(cells_dict, num_points=1000):
    """
    For each species, sample random coordinates within its list of H3 cells.
    """
    candidates = {}
    n_species = 0
    for species_id, cell_list in cells_dict.items():
        points = []
        for _ in range(num_points):
            cell = random.choice(cell_list)
            boundary = h3.h3_to_geo_boundary(cell, geo_json=True)  # List of (lon, lat)
            sampled_point = sample_point_in_polygon(boundary)
            # Normalize from degrees to [-1, 1] range
            lon_norm = sampled_point[0] / 180.0
            lat_norm = sampled_point[1] / 90.0
            points.append([lon_norm, lat_norm])
        candidates[species_id] = points
        n_species += 1
        if n_species % 10 == 0:
            print(f'Species: {n_species}')
    return candidates



def generate_candidates(data_dir, locs, class_ids, class_to_taxa, h3_resolution=1, proximity_k=1):
    """
    Main entry point: generate and save H3-based negatives (cells and points).
    """
    presence_cells = get_h3_cells(class_to_taxa, class_ids, locs, cell_resolution=h3_resolution)
    absence_cells, proximity_cells = gen_negative_cells(presence_cells, h3_resolution=h3_resolution, proximity_k=proximity_k)

    absence_slds_candidates = invert_species_to_cells(absence_cells)
    proximity_slds_candidates = invert_species_to_cells(proximity_cells)

    print('Negatives SLDS: Done')

    # Save cell-based masks
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'presence_h3_cells.json'), 'w') as f:
        json.dump(presence_cells, f)
    with open(os.path.join(data_dir, 'absence_h3_cells.json'), 'w') as f:
        json.dump(absence_cells, f)
    with open(os.path.join(data_dir, 'proximity_h3_cells.json'), 'w') as f:
        json.dump(proximity_cells, f)
    with open(os.path.join(data_dir, 'absence_slds_candidates.json'), 'w') as f:
        json.dump(absence_slds_candidates, f)
    with open(os.path.join(data_dir, 'proximity_slds_candidates.json'), 'w') as f:
        json.dump(proximity_slds_candidates, f)

    # Sample coordinate-based negatives for each species - Training
    absence_ssdl_candidates = sample_points_from_h3_cells(absence_cells, num_points=1000)
    proximity_ssdl_candidates = sample_points_from_h3_cells(proximity_cells, num_points=1000)

    # Sample coordinate-based negatives for each species
    with open(os.path.join(data_dir, 'absence_ssdl_candidates.json'), 'w') as f:
        json.dump(absence_ssdl_candidates, f)
    with open(os.path.join(data_dir, 'proximity_ssdl_candidates.json'), 'w') as f:
        json.dump(proximity_ssdl_candidates, f)

    print("H3-based negatives generated and saved.")



if __name__ == '__main__':
    import datasets

    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    obs_file = os.path.join(data_dir, 'geo_prior_train.csv')
    taxa_file = os.path.join(data_dir, 'geo_prior_train_meta.json')
    taxa_file_subset = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = datasets.get_taxa_of_interest('iucn_taxa', num_aux_species=0,
                                                     taxa_file=taxa_file, taxa_file_subset=taxa_file_subset)
    locs, labels, _, _, _, _ = datasets.load_inat_data(obs_file, taxa_of_interest)
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    generate_candidates(data_dir, locs, class_ids, class_to_taxa, h3_resolution=1, proximity_k=2)

