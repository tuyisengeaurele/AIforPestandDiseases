import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ===============================
# 1. CONFIGURATION
# ===============================
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Make sure this matches the file name you generated in the training step
# If you saved as .h5, change this to "sericulture_v3_best.h5"
MODEL_NAME = "sericulture_v3_best.keras"

# CRITICAL: These must be in ALPHABETICAL ORDER because
# Keras flow_from_directory sorts folders alphabetically by default.
CLASS_NAMES = [
    "disease_free",  # 0
    "grasserie_silkworms",  # 1
    "healthy_silkworms",  # 2
    "leaf_rust",  # 3
    "leaf_spot"  # 4
]

model = None  # global model instance


# ===============================
# 2. LOAD MODEL
# ===============================
def load_trained_model():
    global model
    try:
        if not os.path.exists(MODEL_NAME):
            messagebox.showerror("File Missing",
                                 f"Cannot find model file:\n{MODEL_NAME}\n\nPlease make sure it is in the same folder as this script.")
            return

        # Load the model (compile=False is faster for inference)
        model = load_model(MODEL_NAME, compile=False)
        print(f"[INFO] Model '{MODEL_NAME}' loaded successfully.")

    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Could not load model:\n{e}")
        root.destroy()


# ===============================
# 3. PREPARE IMAGE
# ===============================
def prepare_image_for_model(img_path):
    try:
        # Load and resize the image
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)

        # Normalize (divide by 255) - Must match training!
        img_array = img_array / 255.0

        # Add batch dimension (1, 224, 224, 3)
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
                               "The AI model is not loaded.")
        return

    test_array = prepare_image_for_model(file_path)
    if test_array is None:
        return

    # Run Prediction
    predictions = model.predict(test_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_label = CLASS_NAMES[class_index]

    # Format the text (remove underscores, capitalize)
    display_text = predicted_label.replace('_', ' ').title()

    # Update Label with Color Coding
    if "healthy" in predicted_label or "disease_free" in predicted_label:
        color = "#006400"  # Dark Green for Good
    else:
        color = "#8B0000"  # Dark Red for Bad

    result_label.config(
        text=f"Result: {display_text}\nConfidence: {confidence:.2f}%",
        fg=color,
        font=("Arial", 16, "bold")
    )


# ===============================
# 5. BROWSE IMAGE
# ===============================
def browse_image_file():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        return

    try:
        # Display the image in the GUI
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display only
        img_tk = ImageTk.PhotoImage(img)

        image_display_label.config(image=img_tk)
        image_display_label.image = img_tk

        path_label.config(text=f"Selected: {os.path.basename(file_path)}")

        # Trigger Classification
        classify_image(file_path)

    except Exception as e:
        messagebox.showerror("File Error",
                             f"Could not display image:\n{e}")


# ===============================
# 6. GUI SETUP
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sericulture AI - Model V3")
    root.geometry("450x600")
    root.resizable(False, False)

    # 1. Header
    title_label = tk.Label(
        root,
        text="🐛 Sericulture Disease Detector V3 🍃",
        font=("Helvetica", 16, "bold"),
        pady=15
    )
    title_label.pack()

    # 2. Button
    browse_button = tk.Button(
        root,
        text="Select Image to Analyze",
        command=browse_image_file,
        font=("Arial", 12),
        bg="#007bff",  # Blue button
        fg="white",
        activebackground="#0056b3",
        padx=20,
        pady=10
    )
    browse_button.pack(pady=10)

    # 3. Filename display
    path_label = tk.Label(root, text="No image selected", font=("Arial", 9), fg="gray")
    path_label.pack()

    # 4. Image Placeholder
    image_display_label = tk.Label(root, text="[Image Preview]", bg="#f0f0f0", width=40, height=15)
    image_display_label.pack(pady=10, padx=10)

    # 5. Result Display
    result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
    result_label.pack(pady=20)

    # Load the model after UI is ready
    root.after(500, load_trained_model)

    root.mainloop()