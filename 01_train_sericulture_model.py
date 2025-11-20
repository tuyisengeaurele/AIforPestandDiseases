import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# configuration variables
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
# paths
BASE_DATA_DIR = 'C:/Users/user/OneDrive/Desktop/auris/Notes/Y3/Computing Intelligence and Applications [CE80561]/AIforPestandDiseases/augmented_dataset'
TRAIN_DIR = BASE_DATA_DIR + '/train'
TEST_DIR = BASE_DATA_DIR + '/test'
MODEL_NAME = 'sericulture_disease_and_pest_detector_model.h5'

# data loading and augmentation
print("1. Setting up data generators...")

# data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # normalization
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# rescale only for test
test_datagen = ImageDataGenerator(rescale=1. / 255)

# training
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # testing
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    print(f"Could not load data from directories")
    print(f"Details: {e}")
    exit()

NUM_CLASSES = train_generator.num_classes
print(f"   -> Successfully detected {NUM_CLASSES} classes: {list(train_generator.class_indices.keys())}")

# model building (MobileNetV2 Transfer Learning)
print("\n2. Building MobileNetV2 Model...")

# 1. Loading the base model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# freezing the base layers for the first training phase
base_model.trainable = False

# 2. custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 3. compiling the model (feature extraction)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("   -> Model compiled and ready for training.")

# 3. training phase 1: feature extraction
EPOCHS_PHASE_1 = 10
print(f"\n3. Starting Training Phase 1 ({EPOCHS_PHASE_1} Epochs - New Layers Only)...")

history_1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_1,
    validation_data=test_generator,
    verbose=1
)

# 4. training Phase 2: fine-tuning
EPOCHS_PHASE_2 = 10  # additional 10 epochs for fine-tuning
print(f"\n4. Starting Training Phase 2 ({EPOCHS_PHASE_2} Epochs - Fine-Tuning)...")

# unfreeze the base model
base_model.trainable = True

# re-freeze the bottom 100 layers to preserve low-level features
for layer in base_model.layers[:100]:
    layer.trainable = False

# recompiling the model with an even lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_2,
    initial_epoch=EPOCHS_PHASE_1,  # start epoch counting from where phase 1 left off
    validation_data=test_generator,
    verbose=1
)

# 5. final evaluation and saving
print("\n5. Final Evaluation and Saving...")

# final evaluation
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Final Model Performance:")
print(f"   Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# saving the model
model.save(MODEL_NAME)
print(f"   -> Model saved successfully as '{MODEL_NAME}'")