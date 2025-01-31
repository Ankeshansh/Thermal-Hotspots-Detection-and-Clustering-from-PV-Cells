import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler

import warnings
warnings.filterwarnings('ignore')
 
dataset = "C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\images"
modelPath = "C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\best.pt"
outgraypath = "gray.png"
mask_folder = "C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\mask"
min_pixel_size = 45.0
output_path = "C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\results.csv"

from ThermalStats import inferToCsv, process_thermal_image, calculate_boxwise_temperature_statistics, calculate_temp_stats
from Masks import create_blended_image_with_hotspots, process_images_with_statistics
from Featureextraction import apply_edge_filters, extract_features_from_dataframe

bounding_boxes_df = inferToCsv(dataset, modelPath, outgraypath)

tempstats_df, global_min_temp, global_max_temp = calculate_temp_stats(bounding_boxes_df, dataset)

process_images_with_statistics(dataset, tempstats_df, mask_folder)

filtered_features_df, hotspot_features_array = extract_features_from_dataframe(dataset, mask_folder, tempstats_df, min_pixel_size)

# Modified KMeans Clustering
def mark_and_cluster_hotspots_kmeans(hotspot_features_array, global_min_temp, global_max_temp, n_clusters):
    """
    Marks the hotspots on the image, clusters them using KMeans, and evaluates the clustering.

    Parameters:
    - image: Grayscale or binary image with hotspots.
    - hotspot_features: Features extracted from the hotspots.
    - n_clusters: Number of clusters to use in KMeans.

    Returns:
    - clusters: Cluster assignments for each hotspot.
    - silhouette_score_val: Silhouette score of the clustering.
    """

    # Define the range for weights
    param_distributions = {
        "w1": np.linspace(4.0, 5.0, 10),  # Range for the 1st feature
        "w2": np.linspace(0.5, 1.5, 10),  # Range for the 2nd feature
        "w3": np.linspace(0.5, 1.5, 10),  # Range for the 3rd feature
        "w4": np.linspace(2.0, 5.0, 10),  # Emphasize the 4th feature
        "w5": np.linspace(0.5, 1.5, 10),  # Range for the 5th feature
        "w6": np.linspace(0.5, 1.5, 10),  # Range for the 6th feature
        "w7": np.linspace(0.5, 1.5, 10),  # Range for the 7th feature
        "w8": np.linspace(0.5, 1.5, 10),  # Range for the 8th feature
        "w9": np.linspace(0.5, 1.5, 10),  # Range for the 9th feature
        "w10": np.linspace(1.0, 2.0, 10),  # Range for the 10th feature
    }

    # Generate random weight samples
    n_iter = 50  # Number of random weight combinations to test
    random_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)

    best_score = -1
    best_weights = None

    if len(hotspot_features_array) < n_clusters:
        n_clusters = len(hotspot_features)


    for params in random_sampler:
        # Create the weight vector
        weights = [params["w1"], params["w2"], params["w3"], params["w4"],
           params["w5"], params["w6"], params["w7"], params["w8"],
           params["w9"], params["w10"]]

        # Scale the features by the weights
        weighted_features = hotspot_features_array * weights

        # Custom scaling using max_temp and min_temp
        hotspot_features_scaled = (weighted_features - global_min_temp) / (global_max_temp - global_min_temp)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters).fit(hotspot_features_scaled)

        # Evaluate the clustering using silhouette score
        score = silhouette_score(hotspot_features_scaled, kmeans.labels_)

        # Update the best score and weights if the current setup is better
        if score > best_score:
            best_score = score
            best_weights = weights
            best_kmeans = kmeans
            clusters = kmeans.labels_

    # Display the best weights and silhouette score
    print(f"Best Weights: {best_weights}")
    print(f"Best Silhouette Score: {best_score}")
    
    filtered_features_df['Cluster_id'] = clusters
    filtered_features_df.to_csv(output_path, index=False)

    # Reduce dimensions for visualization using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(hotspot_features_scaled)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_id]
        if cluster_id == -1:
            label = "Noise"
            color = "gray"
        else:
            label = f"Cluster {cluster_id}"
            color = None
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label, cmap="tab20", c=color)

    plt.title(f"KMeans Clustering (Silhouette Score: {best_score:.2f})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()

    return clusters, best_score, best_weights, filtered_features_df

# Specify the number of clusters
n_clusters = 4 # Set based on your dataset
clusters, silhouette_score_val, best_weights, filtered_df = mark_and_cluster_hotspots_kmeans(hotspot_features_array, global_min_temp, global_max_temp, n_clusters=n_clusters)

# The filtered_df dataframe contains the final hotspot features and their cluster ids