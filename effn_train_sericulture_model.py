import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import get_file # Needed for manual weight downloading

# force channels last to fix dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

# 1. configurations
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 16
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 10

# dataset path
TRAIN_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"
MODEL_FILENAME = "sericulture_efficientnetb4_model.keras"

# 2. data generators

print("\n1) Setting up data generators")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.20
)

# training Data
print("Loading training data (80%)")
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# validation Data
print("Loading validation data (20%)")
val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"Classes detected: {train_generator.class_indices}")

# 3. class weights

print("\n2) Calculating class weights")
class_weights_list = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_list))
print(f"Weights applied: {class_weights}")

# 4. Building model

print("\n3) Building EfficientNetB4 model")

# Step A: define the input explicitly
inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Step B: build the architecture with weights=None
base_model = EfficientNetB4(
    weights=None,  # loaded manually in Step C
    include_top=False,
    input_tensor=inputs
)

# Step C: download and load weights manually
# This bypasses the Keras constructor bug causing the shape mismatch
weights_path = get_file(
    "efficientnetb4_notop.h5",
    "https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5",
    cache_subdir="models"
)
print(f"Loading weights from: {weights_path}")
base_model.load_weights(weights_path)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. claabacks

callbacks = [
    ModelCheckpoint(
        MODEL_FILENAME,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
]

# 6. phase 1 training

print(f"\n4) Phase 1: Training head ({EPOCHS_PHASE_1} epochs)")

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_1,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# 7. phase 2 training

print(f"\n5) Phase 2: Fine-tuning ({EPOCHS_PHASE_2} epochs)")

base_model.trainable = True

freeze_until = 400
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
for layer in base_model.layers[freeze_until:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_2,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print(f"\nModel saved as {MODEL_FILENAME}")