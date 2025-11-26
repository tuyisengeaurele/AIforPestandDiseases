import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===========================
# CONFIGURATION
# ===========================
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 10

# Your dataset path
TRAIN_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"
MODEL_FILENAME = "sericulture_v3_best.keras"

# ===========================
# 1. DATA GENERATORS
# ===========================
print("\n1) Setting up Data Generators...")

datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.20
)

# Training Data
print("   - Loading Training Data (80%)...")
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation Data
print("   - Loading Validation Data (20%)...")
val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"   -> Classes Detected: {train_generator.class_indices}")

# ===========================
# 2. CLASS WEIGHTS
# ===========================
print("\n2) Calculating Class Weights...")
class_weights_list = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_list))
print(f"   -> Weights applied: {class_weights}")

# ===========================
# 3. BUILD MODEL
# ===========================
print("\n3) Building MobileNetV2 model...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
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

# ===========================
# 4. CALLBACKS
# ===========================
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

# ===========================
# 5. PHASE 1 TRAINING
# ===========================
print(f"\n4) Phase 1: Training Head ({EPOCHS_PHASE_1} epochs)...")

# FIXED: Removed manual 'steps_per_epoch' to prevent crashing
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_1,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ===========================
# 6. PHASE 2 TRAINING
# ===========================
print(f"\n5) Phase 2: Fine-Tuning ({EPOCHS_PHASE_2} epochs)...")

base_model.trainable = True
freeze_until = 120
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
for layer in base_model.layers[freeze_until:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# FIXED: Removed manual 'steps_per_epoch' here as well
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_2,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print(f"\nDONE. The best version of the model is saved as: {MODEL_FILENAME}")