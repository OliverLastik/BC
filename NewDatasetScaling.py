from PIL import Image
import os


def create_image_tiles(input_dir, output_dir, tile_size=(224, 224), overlap=0):
    """ Process all images in the input directory, slicing them into smaller tiles. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for author in os.listdir(input_dir):
        author_path = os.path.join(input_dir, author)
        author_output_path = os.path.join(output_dir, author)
        if not os.path.exists(author_output_path):
            os.makedirs(author_output_path)

        for filename in os.listdir(author_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(author_path, filename)
                with Image.open(file_path) as img:
                    for i in range(0, img.width, tile_size[0] - overlap):
                        for j in range(0, img.height, tile_size[1] - overlap):
                            # Define the bounding box for each tile
                            box = (i, j, i + tile_size[0], j + tile_size[1])
                            if box[2] <= img.width and box[3] <= img.height:
                                tile = img.crop(box)
                                tile_name = f'{filename[:-4]}_{i}_{j}.png'
                                tile.save(os.path.join(author_output_path, tile_name))


# Example usage
input_dir = r'E:\NonResized'
output_dir = r'E:\TiledImages299'
tile_size = (299, 299)  # or (299, 299) for other models
overlap = 32  # Pixels to overlap tiles

create_image_tiles(input_dir, output_dir, tile_size, overlap)

print(f"Images have been tiled and saved successfully to {output_dir}.")
