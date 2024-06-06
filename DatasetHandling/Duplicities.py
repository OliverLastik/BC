import os


base_dir = r'E:\TiledDataset299'
datasets = ['train', 'val', 'test']


image_sources = {dataset: set() for dataset in datasets}


for dataset in datasets:
    dataset_path = os.path.join(base_dir, dataset)
    for author_folder in os.listdir(dataset_path):
        author_path = os.path.join(dataset_path, author_folder)
        for image_filename in os.listdir(author_path):

            original_id = image_filename.split('_')[0]
            image_sources[dataset].add((author_folder, original_id))

# Check for overlaps
print("Checking for overlaps:")
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        set1 = image_sources[datasets[i]]
        set2 = image_sources[datasets[j]]
        overlap = set1.intersection(set2)
        if overlap:
            print(f"Overlap between {datasets[i]} and {datasets[j]}: {len(overlap)} items")
        else:
            print(f"No overlap between {datasets[i]} and {datasets[j]}")
