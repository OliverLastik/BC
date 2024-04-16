import os
import shutil

# Set the base directories
non_resized_dir = r'E:\DatasetNonResized'
resized_dir = r'E:\resize'
output_dir = r'E:\Dataset'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")


# Function to get the mapping of images to their respective sets
def get_set_mappings(non_resized_dir):
    set_mappings = {'train': {}, 'val': {}, 'test': {}}

    for set_type in ['train', 'val', 'test']:
        set_dir = os.path.join(non_resized_dir, set_type)
        print(f"Processing {set_type} directory...")

        for author in os.listdir(set_dir):
            author_dir = os.path.join(set_dir, author)
            set_mappings[set_type][author] = set(os.listdir(author_dir))
            print(f"Found {len(set_mappings[set_type][author])} images for author {author} in set {set_type}")

    return set_mappings


# Function to copy images based on set mappings
def copy_images_to_sets(resized_dir, output_dir, set_mappings):
    for author in os.listdir(resized_dir):
        author_dir = os.path.join(resized_dir, author)

        for set_type, author_mappings in set_mappings.items():
            if author in author_mappings:
                for image in os.listdir(author_dir):
                    if image in author_mappings[author]:
                        # Set up source and destination paths
                        src_path = os.path.join(author_dir, image)
                        dest_dir = os.path.join(output_dir, set_type, author)

                        # Ensure the destination directory exists
                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                            print(f"Created directory: {dest_dir}")

                        dest_path = os.path.join(dest_dir, image)
                        shutil.copy(src_path, dest_path)
                        print(f"Copied {src_path} to {dest_path}")


# Get the mappings from the non-resized dataset
set_mappings = get_set_mappings(non_resized_dir)

# Copy images from the resized directory to the dataset directory based on the mappings
copy_images_to_sets(resized_dir, output_dir, set_mappings)

print("Images have been copied to the new dataset directory with the same splits.")
