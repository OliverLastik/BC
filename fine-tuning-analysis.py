import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

print("Current working directory:", os.getcwd())
model_path = r'C:\Users\Oliver\OneDrive\Počítač\BP\cdd\pythonProject\best_model_finetuned2.keras'
print("Model exists:", os.path.exists(model_path))

model = tf.keras.models.load_model(model_path)

test_dir = r'E:\Dataset\test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Calculate the correct number of steps per epoch
steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)

predictions = model.predict(test_generator, steps=steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("Class labels:", class_labels)

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Optional: Display some misclassified images
misclassified_idxs = np.where(predicted_classes != true_classes)[0]
if len(misclassified_idxs) > 0:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    for i in np.arange(0, 9):
        if i < len(misclassified_idxs):
            idx = misclassified_idxs[i]
            img = plt.imread(test_generator.filepaths[idx])
            axes[i].imshow(img)
            axes[i].set_title(f"True: {class_labels[true_classes[idx]]}\nPredicted: {class_labels[predicted_classes[idx]]}")
            axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
