import os
import shutil
from PIL import Image, ImageOps

def resize_and_pad_image(input_path, output_path, target_size):
    with Image.open(input_path) as img:
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, target_size, color='black')
        img.save(output_path)

def count_images(directory):
    image_count = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
    return image_count

def main():
    image_directory = input("Enter the directory where your artworks are stored: ")
    if not os.path.isdir(image_directory):
        print("Invalid directory. Please ensure the path is correct.")
        return

    num_images = count_images(image_directory)
    authors = ['EdmundGwerk', 'FerninandHloznik', 'FrantisekStudeny', 'JozefKollar', 'MariaMedvecka', 'MilosBazovsky', 'VincentHloznik']

    if num_images > 7:
        handle_more_than_seven_images(image_directory, authors, num_images)
    else:
        handle_fewer_than_seven_images(image_directory, authors, num_images)

def handle_more_than_seven_images(image_directory, authors, num_images):
    print(f"Handling more than seven images ({num_images} images found).")
    already_organized = input("Are your images already organized into author-specific folders? (yes/no): ").lower()
    if already_organized == 'yes':
        resize_confirm = input("Are all images resized to 299x299? (yes/no): ").lower()
        if resize_confirm != 'yes':
            output_dir = input("Enter the directory where you want to save resized images: ")
            for author in authors:
                author_dir = os.path.join(image_directory, author)
                output_author_dir = os.path.join(output_dir, author)
                if os.path.exists(author_dir):
                    if not os.path.exists(output_author_dir):
                        os.makedirs(output_author_dir)
                    for file in os.listdir(author_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(author_dir, file)
                            output_path = os.path.join(output_author_dir, file)
                            resize_and_pad_image(file_path, output_path, (299, 299))
                            print(f"Resized and saved {file} to {output_path}")
            print("All images have been resized and saved.")
    else:
        print("Please organize your images into folders named by the author and rerun the script.")
        print("Use these folder names for each author:")
        for author in authors:
            print(f"- {author}")

def handle_fewer_than_seven_images(image_directory, authors, num_images):
    print(f"Handling fewer than 7 images ({num_images} images found).")
    for file in os.listdir(image_directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print("Select the author for this image:")
            for i, author in enumerate(authors):
                print(f"{i}: {author}")
            author_index = int(input(f"Enter the number corresponding to the author of {file}: "))
            author_name = authors[author_index]
            author_dir = os.path.join(image_directory, author_name)
            if not os.path.exists(author_dir):
                os.makedirs(author_dir)
            src_path = os.path.join(image_directory, file)
            dst_path = os.path.join(author_dir, file)
            shutil.move(src_path, dst_path)
            print(f"{file} has been moved to {author_dir}")
    print("All images have been organized into their respective author folders.")

if __name__ == "__main__":
    main()
