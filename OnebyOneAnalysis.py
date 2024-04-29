import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import os

# Load the model
model_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\cdd\pythonProject\best_model_finetuned2.keras'
model = tf.keras.models.load_model(model_path)

# Setup test data generator
test_dir = r'E:\ResNetDataset\test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict and report for each image
class_labels = list(test_generator.class_indices.keys())
predictions = model.predict(test_generator, steps=test_generator.samples)
true_classes = test_generator.classes

for idx, pred in enumerate(predictions):
    predicted_class_idx = np.argmax(pred)
    predicted_class = class_labels[predicted_class_idx]
    confidence = np.max(pred)
    image_path = test_generator.filepaths[idx]
    image_name = os.path.basename(image_path)
    true_class_idx = true_classes[idx]
    correct = predicted_class_idx == true_class_idx
    result = "correct" if correct else "incorrect"
    print(f"'{image_name}' artwork belongs to '{predicted_class}' with a confidence of {confidence:.2%} - {result}")
