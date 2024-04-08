import os
import shutil
from sklearn.model_selection import train_test_split

# Set up your paths
base_path = 'C:\\Users\\oliver.lastik\\Desktop\\BP\\dataset'
artists_folders = [folder for folder in os.listdir(base_path) if 'RS' in folder]

# Splitting ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create the train/val/test directories if they don't exist
for split in ['train', 'val', 'test']:
    for artist in artists_folders:
        os.makedirs(os.path.join(base_path, split, artist), exist_ok=True)


# Function to split data and move files
def split_and_move_files(artist_folder):
    files = os.listdir(os.path.join(base_path, artist_folder))
    train_files, test_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=test_ratio / (test_ratio + val_ratio),
                                             random_state=42)

    # Function to move files to their respective directories
    def move_files(files, destination):
        for file in files:
            shutil.move(os.path.join(base_path, artist_folder, file),
                        os.path.join(base_path, destination, artist_folder, file))

    # Move the files
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')


# Apply the function to each artist folder
for artist_folder in artists_folders:
    split_and_move_files(artist_folder)

print("Dataset successfully split into train, validation, and test sets.")


def check_splits(base_path, splits=['train', 'val', 'test']):
    duplicates = {}

    # We will check duplicates for each artist folder
    for artist_folder in os.listdir(base_path):
        # Skip if not a directory
        if not os.path.isdir(os.path.join(base_path, artist_folder)):
            continue

        # Create a set to store filenames for each artist
        all_files = set()

        # Check each split directory for duplicates
        for split in splits:
            split_path = os.path.join(base_path, split, artist_folder)
            # If the split path doesn't exist (e.g., no 'train' folder), skip to next
            if not os.path.exists(split_path):
                continue

            # Iterate over each file in the artist directory
            for file in os.listdir(split_path):
                # Check if file is already in the set
                if file in all_files:
                    # Initialize the duplicate set for this artist if it hasn't been already
                    if artist_folder not in duplicates:
                        duplicates[artist_folder] = set()
                    duplicates[artist_folder].add(file)
                all_files.add(file)

    return duplicates


# Replace with your actual base dataset path
base_dataset_path = 'C:\\Users\\oliver.lastik\\Desktop\\BP\\dataset'
duplicates = check_splits(base_dataset_path)

# Improved printing to show duplicates by artist
if duplicates:
    for artist, files in duplicates.items():
        print(f"Duplicate files found for {artist}: {files}")
else:
    print("No duplicate files found within the same artist's folder. The dataset is properly split.")

