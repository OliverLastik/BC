from PIL import Image
import os


input_dir_path = 'C:\\Users\\Oliver\\Downloads'

output_dir_path = 'C:\\Users\\Oliver\\Downloads\\resize'
target_size = 299

# Make sure the output directory exists, if not, create it
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


def resize_and_pad_image(input_path, output_path, target_size):
    with Image.open(input_path) as img:
        # Resize the image to maintain aspect ratio using the LANCZOS resampling
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        # Create a new image with white background
        background = Image.new('RGB', (target_size, target_size), (255, 255, 255))

        # Paste the resized image onto the center of the background
        background.paste(
            img, (int((target_size - img.width) / 2), int((target_size - img.height) / 2))
        )

        # Save the padded image to the output path
        background.save(output_path)


# Loop through all the images in the input directory and resize them
for file_name in os.listdir(input_dir_path):
    if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
        file_path = os.path.join(input_dir_path, file_name)
        resized_file_path = os.path.join(output_dir_path, file_name)
        resize_and_pad_image(file_path, resized_file_path, target_size)

print(f"All images have been resized and saved to {output_dir_path}.")
