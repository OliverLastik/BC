import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Ending Epoch {epoch}, Logs: {logs}")


train_dir = r'C:\Users\Oliver\OneDrive\Počítač\BP\dataset\train'
validation_dir = r'C:\Users\Oliver\OneDrive\Počítač\BP\dataset\val'
test_dir = r'C:\Users\Oliver\OneDrive\Počítač\BP\dataset\test'

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# InceptionV3 Model as the base
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Initially set the base model to be not trainable

# Custom layers on top
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Initial training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
initial_epochs = 10
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1), ModelCheckpoint('best_model_initial.h5', save_best_only=True)]
)

# Start fine-tuning
base_model.trainable = True  # Unfreeze the base model

# It's important to recompile the model after making any changes to the `trainable` attribute of any layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Use a lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1), ModelCheckpoint('best_model_finetuned.h5', save_best_only=True)]
)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
# Load the best model
model.load_weights('best_model_finetuned.h5')

# Generate predictions for the test set
test_generator.reset()  # Resetting the generator is important before making predictions
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Convert predictions to label indexes
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class labels
true_classes = test_generator.classes

# Get the mapping of classes to indices
class_labels = list(test_generator.class_indices.keys())

# Generate a confusion matrix
conf_mat = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Generate a classification report
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Identify misclassified images
misclassified_idxs = np.where(predicted_classes != true_classes)[0]
print(f"Total misclassified images out of {len(test_generator.classes)}: {len(misclassified_idxs)}")

# Optionally, plot a few misclassified images
for idx in misclassified_idxs[:5]:  # Change the number to see more/less examples
    plt.figure(figsize=(5,5))
    img = test_generator.filepaths[idx]
    img = plt.imread(img)
    true_label = class_labels[true_classes[idx]]
    predicted_label = class_labels[predicted_classes[idx]]
    plt.imshow(img)
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
