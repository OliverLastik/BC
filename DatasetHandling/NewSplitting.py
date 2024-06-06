import os
import shutil
import numpy as np

# Set paths
input_dir_path = r'E:\TiledImages224'
output_base_dir = r'E:\TiledDataset224'

# Ratios for splitting
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create the output base directory if it does not exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)


# Function to split data and copy files
def split_data(source_folder):
    # Group tiles by their source image
    source_groups = {}
    for file in os.listdir(os.path.join(input_dir_path, source_folder)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = file.split('_')[0]  # Assuming filename starts with source image name
            if base_name not in source_groups:
                source_groups[base_name] = []
            source_groups[base_name].append(file)

    all_files = []
    for files in source_groups.values():
        all_files.append(files)  # Now we have a list of lists of files, each list from the same source image

    np.random.shuffle(all_files)  # Shuffle the list of file groups

    # Calculate split indices
    train_end = int(len(all_files) * train_ratio)
    val_end = train_end + int(len(all_files) * val_ratio)

    # Split files into train, validation, test
    train_files = sum(all_files[:train_end], [])
    val_files = sum(all_files[train_end:val_end], [])
    test_files = sum(all_files[val_end:], [])

    # Copy files to their respective directories
    for dest_dir, files in zip(
            [os.path.join(output_base_dir, 'train', source_folder),
             os.path.join(output_base_dir, 'val', source_folder),
             os.path.join(output_base_dir, 'test', source_folder)],
            [train_files, val_files, test_files]):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file in files:
            src_path = os.path.join(input_dir_path, source_folder, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.copy(src_path, dest_path)


# Loop over each author's folder and split the data
for author_folder in os.listdir(input_dir_path):
    if os.path.isdir(os.path.join(input_dir_path, author_folder)):  # Check if it's a directory
        split_data(author_folder)

print("Data has been split and copied successfully.")
