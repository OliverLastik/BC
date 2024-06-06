import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set the directories for the data
data_dir = r'E:\DatasetNonResized'

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
    histograms = {label: [] for label in class_labels}  # Dictionary to store histograms by class

    for class_label in class_labels:
        class_dir = os.path.join(directory, class_label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = cv2.imread(image_path)
            histogram = calc_hist(img)
            data.append(histogram)
            labels.append(class_label)
            histograms[class_label].append(histogram)
    return np.array(data), np.array(labels), class_labels, histograms

# Function to plot average histograms
def plot_histograms(histograms, class_labels):
    color = ('b', 'g', 'r')
    for label in class_labels:
        plt.figure(figsize=(10, 5))
        mean_hist = np.mean(histograms[label], axis=0)  # Average the histograms
        for i, col in enumerate(color):
            plt.plot(mean_hist[i * 256:(i + 1) * 256], color=col, label=f'{col} channel')
        plt.title(f'Average Color Histogram for {label}')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# Create dataset and plot histograms
train_data, train_labels, class_labels, train_histograms = create_dataset(os.path.join(data_dir, 'train'))
plot_histograms(train_histograms, class_labels)
