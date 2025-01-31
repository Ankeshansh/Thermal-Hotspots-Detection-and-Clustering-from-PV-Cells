import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import cv2

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




# Get the hotspot mask for each image
def create_blended_image_with_hotspots(original_image_path, temperature_grid, bounding_boxes, statistics, output_path):
    
    # Load the original thermal image
    original_image = Image.open(original_image_path).convert("RGB")

    # Create overlay arrays
    overlay = np.zeros((512, 640, 3), dtype=np.uint8)
    overlay2 = np.zeros((512,640, 3), dtype=np.uint8)

    # Iterate through each bounding box and its corresponding statistics
    for (x1, y1, x2, y2), stats in zip(bounding_boxes, statistics):
        mean_temp = stats["mean_temperature"]
        std_temp = stats["std_temperature"]

        # Ensure the bounding box is within the bounds of the temperature grid
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, temperature_grid.shape[1]), min(y2, temperature_grid.shape[0])

        # Apply a mask for the hotspots (temperatures above the threshold)
        box_temperatures = temperature_grid[y1:y2, x1:x2]
        if mean_temp is not None and std_temp is not None:
            threshold_temp_min = mean_temp + 1.1 * std_temp

            hotspot_mask = box_temperatures > threshold_temp_min

            # Set overlay color for hotspots (e.g., red)
            overlay[y1:y2, x1:x2][hotspot_mask] = [255, 0, 0]  # Red for hotspots
            overlay2[y1:y2, x1:x2][hotspot_mask] = [255, 255, 255]  # White for hotspots

    # Blend the original image and the overlay
    blended_image = np.array(original_image) * 0.2 + overlay * 0.8  # Adjust blend ratio as needed
    blended_image = blended_image.astype(np.uint8)
    segmented_image = overlay2.astype(np.uint8)

    # Convert blended image to PIL format and save
    segmented_pil_image = Image.fromarray(segmented_image)
    segmented_pil_image.save(output_path)




# Process each image
def process_images_with_statistics(dataset, statistics_df, output_folder):

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image
    # for _, row in tqdm(bounding_boxes_df.iterrows(), total=len(bounding_boxes_df)):
    for image_name in tqdm(statistics_df["image_name"].unique(), total=len(statistics_df["image_name"].unique())):
        # Get the original image path
        original_image_path = os.path.join(dataset, image_name)

        # Load temperature grid for the image
        temperature_grid = process_thermal_image(original_image_path)

        # Get bounding boxes and statistics for the image
        image_data = statistics_df[statistics_df["image_name"] == image_name]
        # print(image_data)

        bounding_boxes = image_data[["x1", "y1", "x2", "y2"]].values.tolist()
        statistics = image_data.to_dict("records")

        # Define output path for the blended image
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_blended.jpg")

        # Create blended image with hotspots
        create_blended_image_with_hotspots(original_image_path, temperature_grid, bounding_boxes, statistics, output_path)

        # print(f"Processed and saved: {output_path}")
