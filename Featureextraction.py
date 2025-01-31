import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label
import os

import flir_image_extractor
path_exif_tool = 'C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\exiftool.exe'

import warnings
warnings.filterwarnings('ignore')


# Get the temperature grid for the thermal image
def process_thermal_image(thermal_image_path):
    """
    Process a thermal image to extract temperature data.
    """
    fir = flir_image_extractor.FlirImageExtractor(exiftool_path=path_exif_tool)
    fir.process_image(thermal_image_path)
    fir.export_thermal_to_csv('temp_thermal_data.csv')  # Temporary file

    # Read the CSV file into a DataFrame
    df = pd.read_csv("temp_thermal_data.csv")

    # Delete the temporary CSV file
    if os.path.exists("temp_thermal_data.csv"):
        os.remove("temp_thermal_data.csv")

    # Convert temperature data into a 2D array to match image dimensions
    temperature_grid = np.zeros((512, 640))
    temperature_grid[df['x'], df['y']] = df['temp (c)']  # Ensure df is properly defined
    return temperature_grid

# Edge Features Extraction
def apply_edge_filters(cropped_region):
    """
    Apply Laplacian, Sobel-X, Sobel-Y filters to a cropped region.

    Parameters:
    - cropped_region: Image region to apply filters (grayscale).

    Returns:
    - edge_features: List of edge-related features.
    """
    # Laplacian filter (second-order derivatives)
    laplacian = cv2.Laplacian(cropped_region, cv2.CV_64F)
    laplacian_response = np.mean(np.abs(laplacian))  # Mean response magnitude

    # Sobel filters (first-order derivatives)
    sobel_x = cv2.Sobel(cropped_region, cv2.CV_64F, 1, 0, ksize=3)  # X direction
    sobel_y = cv2.Sobel(cropped_region, cv2.CV_64F, 0, 1, ksize=3)  # Y direction

    sobel_x_response = np.mean(np.abs(sobel_x))  # Mean gradient in X direction
    sobel_y_response = np.mean(np.abs(sobel_y))  # Mean gradient in Y direction

    # Optional: Canny edge detection (returns edges)
    edges = cv2.Canny(cropped_region, threshold1=50, threshold2=150)
    canny_edge_density = np.sum(edges > 0) / cropped_region.size  # Edge pixel density

    # Combine all edge features
    edge_features = [laplacian_response, sobel_x_response, sobel_y_response, canny_edge_density]
    return edge_features

# Extract the feature vector for all images in the dataset
def extract_features_from_dataframe(image_folder, blended_image_folder, statistics_df, min_pixel_size = 45.0):
    """
    Extract features for each bounding box across images and save the results.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - statistics_csv_path: Path to the CSV file containing bounding box statistics.
    - output_path: Path to save the updated DataFrame with features.

    Returns:
    - updated_df: DataFrame with extracted features.
    """

    # Initialize lists for storing features and additional details
    features = []
    hotspot_features = []

    # Iterate through unique images in the CSV
    for image_name in tqdm(statistics_df["image_name"].unique(), total=len(statistics_df["image_name"].unique())):
        # Construct the image path
        thermal_image_path = os.path.join(image_folder, image_name)
        # print(thermal_image_path)
        image_path = os.path.join(blended_image_folder, image_name.replace('.jpg', '_blended.jpg'))

        # Process the thermal image
        temperature_grid = process_thermal_image(thermal_image_path)  # Replace with your FLIR tool processing
        # print(temperature_grid.shape)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Filter rows for the current image
        image_df = statistics_df[statistics_df['image_name'] == image_name]
        # print(image_df)

        # Iterate through bounding boxes
        for index, row in image_df.iterrows():
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            mean_cell_temp = row['mean_temperature']
            std_dev = row['std_temperature']

            if std_dev >= 1.5:
                # Crop the bounding box region from the temperature grid
                cropped_temp_grid = temperature_grid[y_min:y_max, x_min:x_max]

                # Threshold to isolate hotspots (binary thresholding)
                cropped = image[y_min:y_max, x_min:x_max]
                _, binary = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

                # Label connected components in the binary image (hotspots)
                labeled = label(binary)
                props = regionprops(labeled)

                for prop in props:
                    # Extract shape features
                    aspect_ratio = prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length != 0 else 0
                    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter != 0 else 0
                    compactness = prop.area / prop.convex_area if prop.convex_area != 0 else 0

                    # Extract hotspot coordinates
                    absolute_coords = prop.coords + np.array([y_min, x_min])  # Add offsets

                    # Size and orientation
                    size = prop.area
                    orientation = prop.orientation

                    # Extract the hotspot region temperature from the temperature grid
                    hotspot_pixels = cropped_temp_grid[binary == 255]  # Get hotspot pixels from the grid
                    if len(hotspot_pixels) > 0:
                        mean_hotspot_temp = np.mean(hotspot_pixels)  # Mean temperature of the hotspot region
                    else:
                        mean_hotspot_temp = 0  # If no hotspots, set mean to 0

                    # Calculate temperature sensitivity (hotspot mean - cell mean)
                    temperature_sensitivity = mean_hotspot_temp - mean_cell_temp

                    # Apply edge detection filters
                    edge_features = apply_edge_filters(cropped)

                    # Combine features
                    feature_vector = [
                        aspect_ratio, circularity, compactness, size, orientation,
                        temperature_sensitivity, *edge_features  # Append edge-related features
                    ]
                    hotspot_features.append(feature_vector)

                    # Append the features to the dictionary
                    features.append({
                        "image_name": image_name,
                        "bounding_box_id": index,
                        "absolute_bbox": np.array(prop.bbox) + np.array([y_min, x_min, y_min, x_min]),
                        "absolute_coords": absolute_coords,
                        "AspectRatio": aspect_ratio,
                        "Circularity": circularity,
                        "Compactness": compactness,
                        "Size": size,
                        "Orientation": orientation,
                        "TemperatureSensitivity": temperature_sensitivity,
                        "LaplacianResponse": edge_features[0],
                        "SobelXResponse": edge_features[1],
                        "SobelYResponse": edge_features[2],
                        "CannyEdgeDensity": edge_features[3],
                    })

    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(features)
    
    print(features_df.dtypes)
    print(features_df.head())
    
    # Filter the DataFrame based on the 'Size' column
    filtered_features_df = features_df[features_df["Size"] >= min_pixel_size]

    # Extract feature vectors for the filtered hotspots
    hotspot_features_array = filtered_features_df[
        [
            "AspectRatio",
            "Circularity",
            "Compactness",
            "Size",
            "Orientation",
            "TemperatureSensitivity",
            "LaplacianResponse",
            "SobelXResponse",
            "SobelYResponse",
            "CannyEdgeDensity",
        ]
    ].values

    return filtered_features_df, hotspot_features_array

