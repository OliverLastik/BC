import os

def count_images(directory):
    total_count = 0
    author_counts = {}

    for author in os.listdir(directory):
        author_path = os.path.join(directory, author)
        if os.path.isdir(author_path):
            count = len([name for name in os.listdir(author_path) if name.lower().endswith(('.jpg', '.jpeg', '.png'))])
            author_counts[author] = count
            total_count += count

    return author_counts, total_count

# Specify the directory where the images are stored
directory = r'E:\TiledImages299'

# Get the counts
author_counts, total_count = count_images(directory)

# Print the results
print("Image counts by author:")
for author, count in author_counts.items():
    print(f"{author}: {count}")
print(f"Total images: {total_count}")
