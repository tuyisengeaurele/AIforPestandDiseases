import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd # Added pandas to help with the bar chart data

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# path to dataset
DATA_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"

# trained model file
MODEL_PATH = "sericulture_v3_best.keras"

# Output filenames for the saved images
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
PERFORMANCE_PLOT_FILE = "performance_plot.png"

# 1. load model

print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found.")
    exit()

model = load_model(MODEL_PATH)

# 2. prepare validation data

print("Setting up validation generator")

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.20
)

# shuffle=False is CRITICAL for the Confusion Matrix to work!
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

# 3. run predictions

print("Running predictions on validation set")

y_true = val_generator.classes
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
y_pred = np.argmax(predictions, axis=1)

# 4. generate metrics

print("       DETAILED PERFORMANCE REPORT")

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

print(classification_report(y_true, y_pred, target_names=class_names))

global_f1 = f1_score(y_true, y_pred, average='weighted')
print("-" * 30)
print(f"Global weighted F1-score: {global_f1:.4f}")
print("-" * 30)

# 5. plot & save 1: confusion matrix

print(f"Generating and saving {CONFUSION_MATRIX_FILE}...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion matrix (Global F1: {global_f1:.2f})')
plt.tight_layout()

# save the figure
plt.savefig(CONFUSION_MATRIX_FILE)
print(f" -> Saved confusion matrix to: {os.path.abspath(CONFUSION_MATRIX_FILE)}")
plt.show()

# 6. plot & save 2: performance bar chart

print(f"Generating and saving {PERFORMANCE_PLOT_FILE}...")

# extract F1 scores for plotting
f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

plt.figure(figsize=(10, 6))
# create bar chart
bars = plt.bar(class_names, f1_scores, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0'])

# add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}',
             ha='center', va='bottom')

plt.title('Model F1-score per class')
plt.xlabel('Disease Class')
plt.ylabel('F1 Score (0.0 - 1.0)')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.tight_layout()

# save the figure
plt.savefig(PERFORMANCE_PLOT_FILE)
print(f" -> Saved performance plot to: {os.path.abspath(PERFORMANCE_PLOT_FILE)}")
plt.show()