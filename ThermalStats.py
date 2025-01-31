import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

path_exif_tool = 'C:\\Users\\ansha\\OneDrive\\Desktop\\Reading Thermal\\exiftool.exe'

import flir_image_extractor
import warnings
warnings.filterwarnings('ignore')

# Get the bounding box coordinates for the thermal images
def inferToCsv(folder_path, modelPath, outgraypath):
    bounding_boxes_data = []  # To store bounding box data for all images

    # Load YOLO model
    model2 = YOLO(modelPath)

    # Process each image in the folder
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            bounding_boxes = []
            filtered_bounding_boxes = []

            # Convert the input image to grayscale and save
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(outgraypath, img_gray)

            # Perform inference
            results = model2([outgraypath])

            # Read the original image for visualization
            img_with_boxes = img.copy()

            # Extract bounding box coordinates
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    bounding_boxes.append({"coords": (int(x1), int(y1), int(x2), int(y2)), "area": area})

            # Find the maximum bounding box area
            if bounding_boxes:
                max_area = max(box["area"] for box in bounding_boxes)
                area_threshold = 0.5 * max_area

                # Filter bounding boxes by the area threshold
                for box in bounding_boxes:
                    if box["area"] >= area_threshold:
                        filtered_bounding_boxes.append(box["coords"])
                        # Draw the bounding box on the image
                        x1, y1, x2, y2 = box["coords"]
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Append data for this image
            bounding_boxes_data.append({
                "image_name": image_name,
                "bounding_boxes": filtered_bounding_boxes
            })

    # Save the bounding boxes data to a CSV file
    df = pd.DataFrame(bounding_boxes_data)    
    return df




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




# Get the mean and standard deviation for each bounding box of an image
def calculate_boxwise_temperature_statistics(temperature_grid, bounding_boxes):
    """
    Calculate statistics (mean, std, pixel count) for each bounding box.
    """
    statistics = []

    for idx, box in enumerate(bounding_boxes):
        # Ensure the box is a tuple with four values
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            print(f"Invalid bounding box format: {box}")
            continue  # Skip invalid boxes

        # Unpack the bounding box
        x1, y1, x2, y2 = map(int, box)  # Convert to integers if necessary

        # Ensure bounding box coordinates are within grid bounds
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, temperature_grid.shape[1]), min(y2, temperature_grid.shape[0])

        # Extract temperatures within the bounding box
        box_temperatures = temperature_grid[y1:y2, x1:x2].flatten()

        # Filter out zero values (invalid/missing data)
        valid_temperatures = box_temperatures[box_temperatures > 0]

        # Calculate statistics if there are valid temperatures
        if valid_temperatures.size > 0:
            mean_temp = np.mean(valid_temperatures)
            std_temp = np.std(valid_temperatures)
            num_pixels = valid_temperatures.size
        else:
            mean_temp = None
            std_temp = None
            num_pixels = 0

        # Append the statistics for this bounding box
        statistics.append({
            "bounding_box_id": idx + 1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mean_temperature": mean_temp,
            "std_temperature": std_temp,
            "num_pixels": num_pixels,
        })

    return statistics





def calculate_temp_stats(bounding_boxes_df, dataset):
        
    # Initialize a list to store all statistics
    all_statistics = []
    
    # Initialize variables to track global max and min temperatures
    global_max_temp = float('-inf')
    global_min_temp = float('inf')
    
    # Process each image
    for _, row in tqdm(bounding_boxes_df.iterrows(), total=len(bounding_boxes_df)):
        image_name = row['image_name']
        bounding_boxes = row['bounding_boxes']

        # # Convert string representation of bounding boxes to a list of tuples
        # print(bounding_boxes_str)
        # bounding_boxes = ast.literal_eval(bounding_boxes_str)  # Convert string to list of tuples

        # Image path
        image_path = os.path.join(dataset, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Process the thermal image
        temperature_grid = process_thermal_image(image_path)

        # Update global max and min temperatures
        max_temp = np.max(temperature_grid)
        min_temp = np.min(temperature_grid)
        global_max_temp = max(global_max_temp, max_temp)
        global_min_temp = min(global_min_temp, min_temp)

        # Calculate statistics for the bounding boxes
        statistics = calculate_boxwise_temperature_statistics(temperature_grid, bounding_boxes)

        # Add image name to each statistic
        for stat in statistics:
            stat["image_name"] = image_name
            all_statistics.append(stat)

    # Save the results to a CSV
    output_df = pd.DataFrame(all_statistics)
    print(f"Global Maximum Temperature: {global_max_temp:.2f} °C")
    print(f"Global Minimum Temperature: {global_min_temp:.2f} °C")
    
    return output_df, global_max_temp, global_min_temp
