import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# path
DATA_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"

MODEL_PATH = "sericulture_densenet121_model.keras"

CONFUSION_MATRIX_FILE = "confusion_matrix_densenet.png"
PERFORMANCE_PLOT_FILE = "performance_plot_densenet.png"

# 1. loading the model

print(f"Loading model from {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found.")
    exit()

model = load_model(MODEL_PATH)

# 2. preparing validation data

print("Setting up validation generator")

# DenseNet preprocess_input
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(val_generator.class_indices.keys())
print(f"Classes found: {class_names}")

# 3. running predictions

print("Running predictions on validation set")

y_true = val_generator.classes
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
y_pred = np.argmax(predictions, axis=1)

# 4. metrics

print("MODEL PERFORMANCE (DenseNet121)")

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

print(classification_report(y_true, y_pred, target_names=class_names))

global_f1 = f1_score(y_true, y_pred, average='weighted')
print("-" * 30)
print(f"Global weighted F1-score: {global_f1:.4f}")
print("-" * 30)

# 5. plot & save 1: confusion matrix

print(f"Generating and saving {CONFUSION_MATRIX_FILE}")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'DenseNet121 Confusion Matrix (Global F1: {global_f1:.2f})')
plt.tight_layout()

# saving the figure
plt.savefig(CONFUSION_MATRIX_FILE)
print(f"Saved confusion matrix to: {os.path.abspath(CONFUSION_MATRIX_FILE)}")
plt.show()

# 6. plot & save 2: performance bar chart

print(f"Generating and saving {PERFORMANCE_PLOT_FILE}")

# extracting F1 scores for plotting
f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_scores, color=['#009688', '#4DB6AC', '#00796B', '#004D40', '#80CBC4'])

# adding labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}',
             ha='center', va='bottom')

plt.title('DenseNet121 F1-score per class')
plt.xlabel('Disease Class')
plt.ylabel('F1 Score (0.0 - 1.0)')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.tight_layout()

# saving the figure
plt.savefig(PERFORMANCE_PLOT_FILE)
print(f"Saved performance plot to: {os.path.abspath(PERFORMANCE_PLOT_FILE)}")
plt.show()