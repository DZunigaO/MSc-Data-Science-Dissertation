import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def build_regions(data_dir, n_regions=4, save_path="region_model.json"):
    # Load data
    with open(os.path.join(data_dir, 'n_cells_species.json'), 'r') as f:
        n_cell_dict = json.load(f)
    with open(os.path.join(data_dir, 'kl_divergence.json'), 'r') as f:
        kl_dict = json.load(f)
    with open(os.path.join(data_dir, 'ap_diff.json'), 'r') as f:
        ap_diff_dict = json.load(f)

    df_n_cell = pd.DataFrame(list(n_cell_dict.items()), columns=['taxa', 'n_cells'])
    df_kl = pd.DataFrame(list(kl_dict.items()), columns=['taxa', 'kl'])
    df_diff_ap = pd.DataFrame(list(ap_diff_dict.items()), columns=['taxa', 'ap_diff'])
    df = df_n_cell.merge(df_kl, on='taxa').merge(df_diff_ap, on='taxa')

    n_cells = df['n_cells'].values
    kl = df['kl'].values
    ap_diff = df['ap_diff'].values
    p_abs = 0.5 * (ap_diff + 1)

    # Scale inputs only
    scaler_inputs = StandardScaler()
    inputs_scaled = scaler_inputs.fit_transform(np.column_stack([kl, n_cells]))

    # Cluster
    kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    labels = kmeans.fit_predict(inputs_scaled)

    # Region stats (mean p_abs per cluster)
    region_stats = {}
    for i in range(n_regions):
        mask = labels == i
        region_stats[i] = {'mean_p_abs': float(p_abs[mask].mean())}

    # Save model
    model = {
        "scaler_inputs_mean": scaler_inputs.mean_.tolist(),
        "scaler_inputs_scale": scaler_inputs.scale_.tolist(),
        "centroids": kmeans.cluster_centers_.tolist(),
        "region_stats": region_stats
    }
    with open(os.path.join(data_dir,save_path), "w") as f:
        json.dump(model, f, indent=2)

    print(f"✅ Model saved to {save_path}")
    return df, labels, kmeans, scaler_inputs, region_stats


with open("paths.json", "r") as f:
    paths = json.load(f)
df, labels, kmeans, scaler_inputs, region_stats = build_regions(paths["train"], n_regions=7)





def plot_regions(df, kmeans, scaler_inputs, region_stats, n_points=200):
    kl_grid = np.linspace(df['kl'].min(), df['kl'].max(), n_points)
    nc_grid = np.linspace(df['n_cells'].min(), df['n_cells'].max(), n_points)
    kl_mesh, nc_mesh = np.meshgrid(kl_grid, nc_grid)
    grid_points = np.column_stack([kl_mesh.ravel(), nc_mesh.ravel()])

    # Scale inputs
    grid_scaled = scaler_inputs.transform(grid_points)

    # Assign each grid point to a region
    grid_labels = kmeans.predict(grid_scaled)
    grid_map = grid_labels.reshape(kl_mesh.shape)

    # Plot regions (categorical colors, not values)
    plt.figure(figsize=(10, 10))
    contour = plt.contourf(kl_mesh, nc_mesh, grid_map, cmap="tab10", alpha=0.4)

    # Overlay region labels at centroids
    centroids_scaled = np.array(kmeans.cluster_centers_)[:, :2]
    centroids_orig = centroids_scaled * scaler_inputs.scale_ + scaler_inputs.mean_

    for i, (x, y) in enumerate(centroids_orig):
        value = region_stats[i]['mean_p_abs']
        plt.text(x, y, f"{value:.2f}", color="black", fontsize=20,
                 ha="center", va="center",
                 bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.3"))

    # Bigger axis fonts
    plt.xlabel("KL Divergence", fontsize=18)
    plt.ylabel("N Presence Cells", fontsize=18)
    plt.title("Regions with Mean P Absence", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()




plot_regions(df, kmeans, scaler_inputs, region_stats)
