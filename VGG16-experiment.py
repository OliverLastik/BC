import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path setup
validation_dir = r'E:\TiledDataset224\val'
train_dir = r'E:\TiledDataset224\train'
test_dir = r'E:\TiledDataset224\test'

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

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# VGG16 as the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Initially set the base model to be not trainable

# Adding custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)  # Ensure Flatten is correctly applied
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding a dropout layer
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initial training
initial_epochs = 10
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1), ModelCheckpoint('best_model_initial_vgg16_newdataset.keras', save_best_only=True)]
)

# Start fine-tuning
base_model.trainable = True  # Unfreeze the base model
for layer in base_model.layers[:15]:  # Freeze the first 15 layers
    layer.trainable = False

# Recompile the model with adjusted learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.6e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1), ModelCheckpoint('best_model_finetuned_vgg16_newdataset.keras', save_best_only=True)]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


