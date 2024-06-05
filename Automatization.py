import os
import shutil
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def resize_and_pad_image(input_path, output_path, target_size=(299, 299)):
    with Image.open(input_path) as img:
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, target_size, color='black')
        img.save(output_path)

def organize_images(image_directory, authors):
    num_images = sum([len(files) for r, d, files in os.walk(image_directory)])
    if num_images > 7:
        return handle_more_than_seven_images(image_directory, authors)
    else:
        return handle_fewer_than_seven_images(image_directory, authors)

def handle_more_than_seven_images(image_directory, authors):
    num_images = sum([len(files) for r, d, files in os.walk(image_directory)])
    print(f"Handling more than seven images.({num_images} images)")
    if input("Are your images already organized into author-specific folders? (yes/no): ").lower() == 'no':
        print("Please organize your images into folders named by the author and rerun the script.")
        print("Use these folder names for each author:")
        for author in authors:
            print(f"- {author}")
    confirm = input("Are your images resized to 299x299? (yes/no): ")
    if confirm.lower() == 'no':
        output_dir = input("Enter the directory where you want to save resized images: ")
        for author in authors:
            author_dir = os.path.join(image_directory, author)
            output_author_dir = os.path.join(output_dir, author)
            if not os.path.exists(output_author_dir):
                os.makedirs(output_author_dir)
            if os.path.exists(author_dir):
                for file in os.listdir(author_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        resize_and_pad_image(os.path.join(author_dir, file), os.path.join(output_author_dir, file))
                print(f"Resized and organized images for {author}.")
        print("All images have been resized and saved.")
        print("Identify the author of the artworks.")
        return output_dir
    else:
        print("Identify the author of the artworks.")
        return image_directory


def handle_fewer_than_seven_images(image_directory, authors):
    num_images = sum([len(files) for r, d, files in os.walk(image_directory)])
    print(f"Handling fewer than 7 images.({num_images} images)")
    output_directory = input("Enter the directory where you want to save organized images: ")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

        # Create subfolders for each author
    author_directories = {}
    for author in authors:
        author_dir = os.path.join(output_directory, author)
        os.makedirs(author_dir, exist_ok=True)
        author_directories[author] = author_dir

    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for file in image_files:
        print("Select the author for this image:")
        for i, author in enumerate(authors):
            print(f"{i}: {author}")
        author_index = int(input(f"Enter the number corresponding to the author of {file}: "))
        author_name = authors[author_index]
        author_dir = author_directories[author_name]

        src_path = os.path.join(image_directory, file)
        dst_path = os.path.join(author_dir, file)
        resize_and_pad_image(src_path, dst_path)
        print(f"{file} has been resized and moved to {dst_path}.")
    print("Identify the author of the artworks.")
    return output_directory

def evaluate_model(test_dir):
    model_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\cdd\pythonProject\best_model_finetuned_InceptionV3_with_graphs.keras.keras'
    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(299, 299),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    predictions = model.predict(test_generator, steps=test_generator.samples)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    for idx, pred in enumerate(predictions):
        predicted_class = class_labels[predicted_classes[idx]]
        confidence = np.max(pred)
        image_path = test_generator.filepaths[idx]
        image_name = os.path.basename(image_path)
        correct = predicted_classes[idx] == true_classes[idx]
        result = "correct" if correct else "incorrect"
        print(f"'{image_name}' artwork belongs to '{predicted_class}' with a confidence of {confidence:.2%} - {result}")

    # Ask if the user wants to see the confusion matrix
    if input("Do you want to see the confusion matrix? (yes/no): ").lower() == 'yes':
        display_confusion_matrix(true_classes, predicted_classes, class_labels)


def display_confusion_matrix(true_classes, predicted_classes, class_labels):
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def main():
    authors = ['EdmundGwerk', 'FerninandHloznik', 'FrantisekStudeny', 'JozefKollar', 'MariaMedvecka', 'MilosBazovsky', 'VincentHloznik']
    image_directory = input("Enter the directory where your artworks are stored: ")
    output_directory = organize_images(image_directory, authors)
    evaluate_model(output_directory)


if __name__ == "__main__":
    main()
