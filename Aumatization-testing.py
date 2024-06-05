import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def load_and_evaluate(test_dir):
    # Pre-defined model path
    model_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\cdd\pythonProject\best_model_finetuned2.keras'
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

    return true_classes, predicted_classes, class_labels

def display_confusion_matrix(true_classes, predicted_classes, class_labels):
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main():
    test_dir = input("Enter the directory where your test images are stored: ")

    true_classes, predicted_classes, class_labels = load_and_evaluate(test_dir)

    show_matrix = input("Do you want to see the confusion matrix? (yes/no): ")
    if show_matrix.lower() == 'yes':
        display_confusion_matrix(true_classes, predicted_classes, class_labels)

if __name__ == "__main__":
    main()
