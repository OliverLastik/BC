import os
import shutil
import numpy as np

# Set paths
input_dir_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\NotResizedDataset'
output_base_dir = r'E:\DatasetNonResized'

# Ratios for splitting
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create the output base directory if it does not exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Function to split data and copy files
def split_data(author_folder):
    # Paths for train, validation, test
    train_dir = os.path.join(output_base_dir, 'train', author_folder)
    val_dir = os.path.join(output_base_dir, 'val', author_folder)
    test_dir = os.path.join(output_base_dir, 'test', author_folder)

    # Create directories if they do not exist
    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Get all images
    images = [file for file in os.listdir(os.path.join(input_dir_path, author_folder))
              if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(images)  # Shuffle the images randomly

    # Calculate split indices
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)

    # Split images into train, validation, test
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Function to copy files
    def copy_files(files, dest_dir):
        for file in files:
            src_path = os.path.join(input_dir_path, author_folder, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.copy(src_path, dest_path)

    # Copy files to their respective directories
    copy_files(train_images, train_dir)
    copy_files(val_images, val_dir)
    copy_files(test_images, test_dir)
    print(f"Finished splitting data for {author_folder}")

# Loop over each author's folder and split the data
for author_folder in os.listdir(input_dir_path):
    if os.path.isdir(os.path.join(input_dir_path, author_folder)):  # Check if it's a directory
        split_data(author_folder)

print("Data has been split and copied successfully.")
