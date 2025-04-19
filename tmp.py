import json
import math
import os
import zipfile
from datetime import datetime

def process_camera_file(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r') as f:
        cameras = json.load(f)

    # Ensure we have at least 10 cameras
    if len(cameras) < 10:
        print(f"Warning: {input_file} contains fewer than 10 cameras. Skipping.")
        return False

    # Calculate the step size to select 10 equally spaced cameras
    step = math.ceil(len(cameras) / 10)

    # Select 10 cameras equally spaced
    selected_cameras = cameras[::step][:10]

    # If we have less than 10 cameras due to rounding, add the last ones
    while len(selected_cameras) < 10:
        selected_cameras.append(cameras[len(selected_cameras) * step])

    # Rename the IDs from 0 to 9
    for new_id, camera in enumerate(selected_cameras):
        camera['id'] = new_id

    # Save the result to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(selected_cameras, f, indent=2)

    print(f"Selected and renamed 10 cameras from {input_file}. Saved to {output_file}")
    return True

# List of input files
# scenes = ["006", "026", "090", "105", "108", "134", "150", "181"]
scenes = ["006"]

# List to store successfully generated output files
generated_files = []

# Process each file
for scene in scenes:
    input_file = f"/home/ml4u/BKTeam/TheHiep/street_gaussians/output/waymo_exp/waymo_val_{scene}/cameras.json"
    if os.path.exists(input_file):
        output_file = f"cameras_10_{scene}.json"
        if process_camera_file(input_file, output_file):
            generated_files.append(output_file)
    else:
        print(f"Warning: {input_file} not found. Skipping.")

print("Processing complete.")

# Zip the generated files
# if generated_files:
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     zip_filename = f"cameras_10_files_{timestamp}.zip"
    
#     with zipfile.ZipFile(zip_filename, 'w') as zipf:
#         for file in generated_files:
#             zipf.write(file)
#             os.remove(file)  # Remove the JSON file after adding it to the zip
    
#     print(f"All generated files have been zipped into {zip_filename}")
#     print("Individual JSON files have been removed.")
# else:
#     print("No files were generated to zip.")