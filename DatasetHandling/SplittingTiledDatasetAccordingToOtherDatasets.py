import os
import shutil
import numpy as np


source_dir_path = r'E:\TiledDataset224'
input_dir_path = r'E:\TiledImages299'
output_base_dir = r'E:\TiledDataset299'

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)


# Function to replicate data split from source to target
def replicate_data_split(source_folder):
    for subset in ['train', 'val', 'test']:
        src_dir = os.path.join(source_dir_path, subset, source_folder)
        dest_dir = os.path.join(output_base_dir, subset, source_folder)

        # Ensure destination directory exists
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # List all files in the source subset directory
        files = os.listdir(src_dir)

        # Copy each file from the input directory to the correct subset in the output directory
        for file in files:
            base_name = '_'.join(file.split('_')[:2])  # Assuming filename is like original_tile_coord
            matching_files = [f for f in os.listdir(os.path.join(input_dir_path, source_folder))
                              if f.startswith(base_name)]

            for matching_file in matching_files:
                src_path = os.path.join(input_dir_path, source_folder, matching_file)
                dest_path = os.path.join(dest_dir, matching_file)
                shutil.copy(src_path, dest_path)


# Loop over each author's folder and replicate the data split
for author_folder in os.listdir(input_dir_path):
    if os.path.isdir(os.path.join(input_dir_path, author_folder)):  # Check if it's a directory
        replicate_data_split(author_folder)

print("Data has been split and copied successfully.")
