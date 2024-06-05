import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize

print("Current working directory:", os.getcwd())
model_path = r'E:\BP\cdd\pythonProject\best_model_finetuned_InceptionV3_with_graphs.keras'
print("Model exists:", os.path.exists(model_path))
model = tf.keras.models.load_model(model_path)

test_dir = r'E:\Dataset/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

try:
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Calculate the correct number of steps per epoch
    steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)

    # Disable multiprocessing
    predictions = model.predict(test_generator, steps=steps_per_epoch, use_multiprocessing=False, workers=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    print("Class labels:", class_labels)

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Generating ROC curve and AUC
    y_pred_proba = predictions  # Model predictions

    # Binarize the true classes
    y_true_binarized = label_binarize(true_classes, classes=range(test_generator.num_classes))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(test_generator.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for all classes
    plt.figure()
    for i in range(test_generator.num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Trieda {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
