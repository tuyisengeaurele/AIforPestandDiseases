import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# path to test
TEST_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\test"

# model
MODEL_PATH = "sericulture_disease_and_pest_detector_model_v2.h5"

# 1. loading the model

print(f"Loading model from {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

model = load_model(MODEL_PATH)

# 2. loading test data

print(f"Reading test images from: {TEST_DIR}")

if not os.path.exists(TEST_DIR):
    print(f"Error: Test directory '{TEST_DIR}' does not exist.")
    exit()

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(f"Classes found: {class_names}")

# 3. running predictions

print("Running predictions on test set")

# true labels
y_true = test_generator.classes

# model predictions
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_pred = np.argmax(predictions, axis=1)

# 4. metrics

print("MODEL PERFORMANCE")

#  report
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_text = classification_report(y_true, y_pred, target_names=class_names)
print(report_text)

# global F1
global_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Global Weighted F1 Score: {global_f1:.4f}")

# 5. plot & save 1: confusion matrix

print("\nSaving Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'V2 Confusion Matrix (Test Data)\nGlobal F1: {global_f1:.2f}')
plt.tight_layout()

# saving the image
plt.savefig('v2_external_test_confusion_matrix.png')
print("Saved as 'v2_external_test_confusion_matrix.png'")
plt.show()

# 6. plot & save 1: class performance

print("\nSaving performance plot")

# extracting F1 scores for each class
f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_scores, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0'])

# Adding value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

plt.title('V2 Model Performance per Class (F1 Score)')
plt.xlabel('Class')
plt.ylabel('F1 Score (0-1)')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.tight_layout()

# Saving the image
plt.savefig('v2_external_test_class_performance.png')
print("Saved as 'v2_external_test_class_performance.png'")
plt.show()