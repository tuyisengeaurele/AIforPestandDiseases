import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import os

IMG_WIDTH = 224
IMG_HEIGHT = 224

MODEL_NAME = "sericulture_densenet121_model.keras"

# path
DATASET_PATH = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"

model = None
CLASS_NAMES = []


def get_classes_dynamically():
    global CLASS_NAMES
    try:
        # Checking if dataset folder exists
        if not os.path.exists(DATASET_PATH):
            messagebox.showerror("Path Error",
                                 f"Could not find dataset folder at:\n{DATASET_PATH}\n\nCannot load class names.")
            return []

        # sub-folders
        classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        classes.sort()

        print(f"Detected {len(classes)} classes: {classes}")
        return classes

    except Exception as e:
        messagebox.showerror("Class load error", f"Error reading dataset folders:\n{e}")
        return []


# 3. loading the model

def load_trained_model():
    global model, CLASS_NAMES

    # 1. load classes first
    CLASS_NAMES = get_classes_dynamically()
    if not CLASS_NAMES:
        root.destroy()
        return

    # 2. load model
    try:
        if not os.path.exists(MODEL_NAME):
            messagebox.showerror("File missing",
                                 f"Cannot find model file:\n{MODEL_NAME}\n\nPlease make sure it is in the same folder as this script.")
            return

        model = load_model(MODEL_NAME, compile=False)
        print(f"Model '{MODEL_NAME}' loaded successfully.")

    except Exception as e:
        messagebox.showerror("Model error",
                             f"Could not load model:\n{e}")
        root.destroy()


# 4. prepare image

def prepare_image_for_model(img_path):
    try:
        # Load and resize
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to array
        img_array = image.img_to_array(img)

        # Expand dimensions to create a batch of 1
        img_array = np.expand_dims(img_array, axis=0)

        # Use DenseNet specific preprocessing
        # DenseNet expects values scaled between 0 and 1 with specific normalization.
        # The preprocess_input function handles this automatically.
        img_array = preprocess_input(img_array)

        return img_array
    except Exception as e:
        messagebox.showerror("Processing error", f"Error preparing image:\n{e}")
        return None


# 5. run prediction

def classify_image(file_path):
    if model is None:
        messagebox.showwarning("Model not loaded", "The model is not loaded.")
        return

    if not CLASS_NAMES:
        messagebox.showwarning("Error", "Class names not loaded.")
        return

    test_array = prepare_image_for_model(file_path)
    if test_array is None:
        return

    # run prediction
    predictions = model.predict(test_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_label = CLASS_NAMES[class_index]

    display_text = predicted_label.replace('_', ' ').title()

    # Color Logic
    if "healthy" in predicted_label.lower() or "disease_free" in predicted_label.lower():
        color = "#006400"  # Green
    else:
        color = "#8B0000"  # Red

    result_label.config(
        text=f"Result: {display_text}\nConfidence: {confidence:.2f}%",
        fg=color,
        font=("Arial", 16, "bold")
    )


# 6. browse image

def browse_image_file():
    file_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_display_label.config(image=img_tk)
        image_display_label.image = img_tk

        path_label.config(text=f"Selected: {os.path.basename(file_path)}")
        classify_image(file_path)

    except Exception as e:
        messagebox.showerror("File Error", f"Could not display image:\n{e}")


# 7. gui setup

if __name__ == "__main__":
    root = tk.Tk()
    # CHANGED: Title for DenseNet121
    root.title("Sericulture Pest and Disease Detector (DenseNet121)")
    root.geometry("450x600")
    root.resizable(False, False)

    # Header
    title_label = tk.Label(
        root,
        # CHANGED: Label for DenseNet121
        text="Sericulture DenseNet121 Detector",
        font=("Helvetica", 16, "bold"),
        pady=15
    )
    title_label.pack()

    # Button
    browse_button = tk.Button(
        root,
        text="Select image",
        command=browse_image_file,
        font=("Arial", 12),
        bg="#009688", # Changed button color to Teal (standard DenseNet color theme)
        fg="white",
        activebackground="#00796B",
        padx=20,
        pady=10
    )
    browse_button.pack(pady=10)

    # Labels
    path_label = tk.Label(root, text="No image selected", font=("Arial", 9), fg="gray")
    path_label.pack()

    image_display_label = tk.Label(root, text="[Image Preview]", bg="#f0f0f0", width=200, height=200)
    image_display_label.pack(pady=10, padx=10)

    result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
    result_label.pack(pady=20)

    # Initialize
    root.after(500, load_trained_model)
    root.mainloop()