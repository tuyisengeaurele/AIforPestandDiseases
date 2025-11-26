import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

IMG_WIDTH = 224
IMG_HEIGHT = 224

MODEL_NAME = "sericulture_disease_and_pest_detector_model_v2.h5"

# path
TRAIN_DIR = r"C:\Users\user\OneDrive\Desktop\auris\Notes\Y3\Computing Intelligence and Applications [CE80561]\AIforPestandDiseases\new_dataset"

model = None
CLASS_NAMES = []


# 2. class loading

def load_class_names():
    global CLASS_NAMES
    if not os.path.exists(TRAIN_DIR):
        messagebox.showerror("Configuration Error",
                             f"Training directory not found:\n{TRAIN_DIR}\n\nCannot load class names.")
        return False

    # subfolders
    try:
        CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR)
                              if os.path.isdir(os.path.join(TRAIN_DIR, d))])

        print(f"Detected classes: {CLASS_NAMES}")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read classes from folder:\n{e}")
        return False


# 3. loading the model

def load_trained_model():
    global model
    try:
        model = load_model(MODEL_NAME, compile=False)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Could not load model:\n{e}\nMake sure the model file exists.")
        root.destroy()


# 4. preparing image

def prepare_image_for_model(img_path):
    try:
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        messagebox.showerror("Processing error",
                             f"Error preparing image:\n{e}")
        return None


# 5. running prediction

def classify_image(file_path):
    if model is None:
        messagebox.showwarning("Model not loaded",
                               "The model was not loaded correctly.")
        return

    if not CLASS_NAMES:
        messagebox.showwarning("Classes not loaded",
                               "Class names could not be loaded from the dataset folder.")
        return

    test_array = prepare_image_for_model(file_path)
    if test_array is None:
        return

    predictions = model.predict(test_array)
    class_index = np.argmax(predictions[0])

    # Get name dynamically
    class_name = CLASS_NAMES[class_index]
    confidence = predictions[0][class_index] * 100

    result_label.config(
        text=f"Prediction: {class_name.replace('_', ' ').title()}\nConfidence: {confidence:.2f}%",
        fg="#004d00",
        font=("Arial", 14, "bold")
    )


# 6. browsing the image

def browse_image_file():
    file_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_display_label.config(image=img_tk)
        image_display_label.image = img_tk

        path_label.config(text=f"Selected: {file_path.split('/')[-1]}")

        classify_image(file_path)

    except Exception as e:
        messagebox.showerror("File error",
                             f"Could not display image:\n{e}")


# 7. gui setup

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sericulture Pests & Disease Detector - V2")
    root.geometry("400x600")
    root.resizable(False, False)

    if load_class_names():
        load_trained_model()
    else:
        root.destroy()

    title_label = tk.Label(
        root,
        text="🦋 Sericulture & Leaf Disease Detector V2 🌿",
        font=("Arial", 16, "bold"),
        pady=10
    )
    title_label.pack()

    browse_button = tk.Button(
        root,
        text="Select image",
        command=browse_image_file,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white",
        pady=5
    )
    browse_button.pack(pady=10)


    path_label = tk.Label(root, text="No image selected", font=("Arial", 10), fg="gray")
    path_label.pack()

    # image preview
    image_display_label = tk.Label(root, borderwidth=1, relief="solid")
    image_display_label.pack(pady=10)

    # result text
    result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
    result_label.pack(pady=10)

    root.mainloop()