import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Path to your model
model_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\cdd\pythonProject\best_model_finetuned2.keras'
model = load_model(model_path)

# Directory containing images
image_dir = r'C:\Users\Oliver\OneDrive\Počítač\BP\TestingImages'
save_dir = r'C:\Users\Oliver\OneDrive\Počítač\BP\TestingImagesResized'

# Class labels
class_labels = ['EdmundGwerk', 'FerninandHloznik', 'FrantisekStudeny', 'JozefKollar', 'MariaMedvecka', 'MilosBazovsky', 'VincentHloznik']


def process_and_predict(image_path):
    print(f"Processing {os.path.basename(image_path)}...")
    # Load and process the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Save the preprocessed image
    img.save(os.path.join(save_dir, os.path.basename(image_path)))

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)
    predicted_class = class_labels[predicted_class_idx[0]]
    confidence = np.max(predictions) * 100

    return f"For image '{os.path.basename(image_path)}', this artwork is most likely by {predicted_class} with a confidence of {confidence:.2f}%."

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Iterate over all images in the directory
for file in os.listdir(image_dir):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, file)
        result = process_and_predict(image_path)
        print(result)
