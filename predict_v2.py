import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ===============================
# 1. CONFIG (MATCH TRAINING)
# ===============================
IMG_WIDTH = 224
IMG_HEIGHT = 224

MODEL_NAME = "best_sericulture_and_leaf_disease_detector_v2.h5"

CLASS_NAMES = [
    "grasserie_silkworms",
    "healthy_silkworms",
    "disease_free",
    "leaf_rust",
    "leaf_spot"
]

model = None  # global model instance


# ===============================
# 2. LOAD MODEL
# ===============================
def load_trained_model():
    global model
    try:
        model = load_model(MODEL_NAME, compile=False)
        print(f"[INFO] Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Could not load model:\n{e}\nMake sure the model file exists.")
        root.destroy()


# ===============================
# 3. PREPARE IMAGE
# ===============================
def prepare_image_for_model(img_path):
    try:
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        messagebox.showerror("Processing Error",
                             f"Error preparing image:\n{e}")
        return None


# ===============================
# 4. RUN PREDICTION
# ===============================
def classify_image(file_path):
    if model is None:
        messagebox.showwarning("Model Not Loaded",
                               "The AI model was not loaded correctly.")
        return

    test_array = prepare_image_for_model(file_path)
    if test_array is None:
        return

    predictions = model.predict(test_array)
    class_index = np.argmax(predictions[0])
    class_name = CLASS_NAMES[class_index]
    confidence = predictions[0][class_index] * 100

    result_label.config(
        text=f"Prediction: {class_name.replace('_', ' ').title()}",
        fg="#004d00",
        font=("Arial", 14, "bold")
    )


# ===============================
# 5. BROWSE IMAGE
# ===============================
def browse_image_file():
    file_path = filedialog.askopenfilename(
        title="Select Image",
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
        messagebox.showerror("File Error",
                             f"Could not display image:\n{e}")


# ===============================
# 6. GUI SETUP
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sericulture & Leaf Disease Detector - V2")
    root.geometry("400x550")
    root.resizable(False, False)

    load_trained_model()

    # Title
    title_label = tk.Label(
        root,
        text="🦋 Sericulture & Leaf Disease Detector V2 🌿",
        font=("Arial", 16, "bold"),
        pady=10
    )
    title_label.pack()

    # Browse button
    browse_button = tk.Button(
        root,
        text="Select Image",
        command=browse_image_file,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white",
        pady=5
    )
    browse_button.pack(pady=10)

    # File path label
    path_label = tk.Label(root, text="No image selected", font=("Arial", 10), fg="gray")
    path_label.pack()

    # Image preview
    image_display_label = tk.Label(root, borderwidth=1, relief="solid")
    image_display_label.pack(pady=10)

    # Result text
    result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
    result_label.pack(pady=10)

    root.mainloop()
