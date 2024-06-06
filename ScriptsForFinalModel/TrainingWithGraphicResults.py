import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle

class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Ending Epoch {epoch}, Logs: {logs}")


validation_dir = r'E:\Dataset\val'
train_dir = r'E:\Dataset\train'
test_dir = r'E:\Dataset\test'

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=batch_size,
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
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1), ModelCheckpoint('best_model_initial_InceptionV3_with_graphs.keras', save_best_only=True)]
)

# Save the initial training history
with open('initial_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Start fine-tuning
base_model.trainable = True  # Unfreeze the base model

# It's important to recompile the model after making any changes to the `trainable` attribute of any layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3.2e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1), ModelCheckpoint('best_model_finetuned_InceptionV3_with_graphs.keras', save_best_only=True)]
)

# Save the fine-tuning history
with open('fine_tuning_history.pkl', 'wb') as file:
    pickle.dump(history_fine.history, file)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Load the histories and plot the training/validation accuracy and loss
with open('initial_training_history.pkl', 'rb') as file:
    initial_history = pickle.load(file)

with open('fine_tuning_history.pkl', 'rb') as file:
    fine_tuning_history = pickle.load(file)

# Combine the histories
acc = initial_history['accuracy'] + fine_tuning_history['accuracy']
val_acc = initial_history['val_accuracy'] + fine_tuning_history['val_accuracy']
loss = initial_history['loss'] + fine_tuning_history['loss']
val_loss = initial_history['val_loss'] + fine_tuning_history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Trénovacia presnosť')
plt.plot(epochs, val_acc, 'r', label='Validačná presnosť')
plt.title('Trénovacia a validačná presnosť')
plt.xlabel('Počet epoch')
plt.ylabel('Presnosť')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Trénovacia strata')
plt.plot(epochs, val_loss, 'r', label='Validačná strata')
plt.title('Trénovacia a validačná strata')
plt.xlabel('Počet epoch')
plt.ylabel('Strata')
plt.legend()

plt.savefig('training_validation_plot.png')
plt.show()
