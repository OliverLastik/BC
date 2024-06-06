import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import cv2
import os

# Set the directories for the data
train_dir = r'E:\DatasetNonResized\train'
val_dir = r'E:\DatasetNonResized\val'
test_dir = r'E:\DatasetNonResized\test'

# Function to calculate color histogram
def calc_hist(img):
    histogram = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
    histogram = np.concatenate(histogram)
    histogram = histogram.flatten()
    return histogram

# Function to load images and calculate histograms
def create_dataset(directory):
    data = []
    labels = []
    class_labels = os.listdir(directory)

    for class_label in class_labels:
        class_dir = os.path.join(directory, class_label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = cv2.imread(image_path)
            histogram = calc_hist(img)
            data.append(histogram)
            labels.append(class_label)
    return np.array(data), np.array(labels), class_labels


# Create datasets
train_data, train_labels, class_labels = create_dataset(train_dir)
val_data, val_labels, _ = create_dataset(val_dir)
test_data, test_labels, _ = create_dataset(test_dir)

# Encode labels
label_to_num = {v: k for k, v in enumerate(class_labels)}
train_labels_num = np.array([label_to_num[label] for label in train_labels])
val_labels_num = np.array([label_to_num[label] for label in val_labels])
test_labels_num = np.array([label_to_num[label] for label in test_labels])


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_data, train_labels_num)

# Validate the model
val_predictions = clf.predict(val_data)
print("\nValidating the model\n")
print(classification_report(val_labels_num, val_predictions, target_names=class_labels))

# Test the model
test_predictions = clf.predict(test_data)
print("Testing the model\n")
print(classification_report(test_labels_num, test_predictions, target_names=class_labels))




