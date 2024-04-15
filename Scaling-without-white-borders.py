from PIL import Image, ImageOps
import os

input_dir_path = r'E:\NonResized'
output_dir_path = r'E:\resize'
target_size = (299, 299)  # This is now a tuple

# Ensure the output directory exists
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


def resize_and_pad_image(input_path, output_path, target_size):
    with Image.open(input_path) as img:
        # Resize the image with the same aspect ratio and ensure one side is 299 pixels
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Pad the image to be 299x299
        img = ImageOps.pad(img, target_size, color='black')

        img.save(output_path)


# Walk through the input directory, and process files from each subdirectory
for subdir, dirs, files in os.walk(input_dir_path):
    for file_name in files:
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full file path
            file_path = os.path.join(subdir, file_name)

            # Construct a new output path by replacing the input_dir_path with output_dir_path in the subdir path
            output_subdir = subdir.replace(input_dir_path, output_dir_path)

            # Ensure the new output subdirectory exists
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Construct the full output file path
            resized_file_path = os.path.join(output_subdir, file_name)

            # Resize and pad the image
            resize_and_pad_image(file_path, resized_file_path, target_size)

print(f"All images have been resized and saved to {output_dir_path}.")
