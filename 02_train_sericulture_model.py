import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 10

# base dataset containing training images
TRAIN_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"

MODEL_NAME = "sericulture_disease_and_pest_detector_model_v2.h5"

# 1) detecting source folders

print("\n1) Detecting class folders")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

# scan directory for classes
CLASSES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

if not CLASSES:
    raise FileNotFoundError("No class folders found in the training directory!")

print(f"Found {len(CLASSES)} classes: {CLASSES}")

# 2) data generators

print("\n2) Setting up ImageDataGenerator")

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

NUM_CLASSES = train_generator.num_classes
print(f"Detected {NUM_CLASSES} classes: {train_generator.class_indices}")

# 3) building model (MobileNetV2)

print("\n3) Building MobileNetV2 model")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
base_model.trainable = False  # freeze for phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4) callbacks

callbacks = [
    ModelCheckpoint(MODEL_NAME, monitor='accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='loss', patience=7, restore_best_weights=True, verbose=1)
]

# 5) phase 1 training (feature extraction)

print(f"\n4) Phase 1 training: {EPOCHS_PHASE_1} epochs")

history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_1,
    callbacks=callbacks,
    verbose=1
)

# 6) phase 2 training (fine-tuning)

print(f"\n5) Phase 2 fine-tuning: {EPOCHS_PHASE_2} epochs")

base_model.trainable = True
freeze_until = 100
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
for layer in base_model.layers[freeze_until:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_2,
    initial_epoch=EPOCHS_PHASE_1,
    callbacks=callbacks,
    verbose=1
)

# 7) saving the final model

model.save(MODEL_NAME)
print(f"Model saved successfully as {MODEL_NAME}")