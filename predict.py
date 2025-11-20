import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- 1. Configuration (MUST MATCH TRAINING!) ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
MODEL_NAME = 'sericulture_pest_detector_model.h5'

# IMPORTANT: These names must be in the exact order the model learned (alphabetical order of folders)
CLASS_NAMES = ['bad_silkworms', 'disease_free', 'good_silkworms', 'leaf_rust', 'leaf_spot']

# Global variable to hold the loaded model
model = None


# --- 2. Model Loading Function ---
def load_trained_model():
    """Loads the Keras model once when the application starts."""
    global model
    try:
        model = load_model(MODEL_NAME, compile=False)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Failed to load model: {e}\nEnsure '{MODEL_NAME}' is in the current directory.")
        # Quit if model fails to load
        root.destroy()


# --- 3. Image Preprocessing Function ---
def prepare_image_for_model(img_path):
    """Loads, resizes, normalizes, and reshapes the image for the model."""
    try:
        # Load the image and resize it
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to a NumPy array
        img_array = image.img_to_array(img)

        # Normalize (rescale to 0-1 range)
        img_array = img_array / 255.0

        # Add the "batch dimension" (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error preparing image: {e}")
        return None


# --- 4. Prediction Logic ---
def classify_image(file_path):
    """Runs the prediction on the selected image file."""
    if model is None:
        messagebox.showwarning("Warning", "AI model not loaded.")
        return

    # Preprocess
    test_image_array = prepare_image_for_model(file_path)
    if test_image_array is None:
        return

    # Predict
    predictions = model.predict(test_image_array)

    # Interpret results
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    predicted_class = CLASS_NAMES[predicted_class_index]

    # Format and display the result
    result_text = (
        f"Prediction: {predicted_class.replace('_', ' ').title()}\n"
        f"Confidence: {confidence:.2f}%"
    )
    result_label.config(text=result_text, fg="#004d00", font=('Arial', 14, 'bold'))


# --- 5. GUI Interface Function ---
def browse_image_file():
    """Opens a file explorer dialog and handles the selected image."""
    # Open the file selection dialog
    file_path = filedialog.askopenfilename(
        title="Select Sericulture Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # Display the image in the GUI
    try:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        image_display_label.config(image=img_tk)
        image_display_label.image = img_tk  # Keep a reference to prevent garbage collection

        path_label.config(text=f"Selected: {file_path.split('/')[-1]}")

        # Run the classification
        classify_image(file_path)

    except Exception as e:
        messagebox.showerror("File Error", f"Could not process the selected file: {e}")


# --- 6. Main Application Setup ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Sericulture AI Predictor")
    root.geometry("400x550")
    root.resizable(False, False)

    # 1. Load Model at Startup
    load_trained_model()

    # 2. Title Label
    title_label = tk.Label(root, text="🌿 Sericulture Disease Detector 🦠", font=('Arial', 16, 'bold'), pady=10)
    title_label.pack()

    # 3. Browse Button
    browse_button = tk.Button(
        root,
        text="Click to Select Image",
        command=browse_image_file,
        font=('Arial', 12),
        bg="#4CAF50",
        fg="white",
        pady=5
    )
    browse_button.pack(pady=10)

    # 4. Path Label (Display selected file name)
    path_label = tk.Label(root, text="No image selected", font=('Arial', 10), fg='gray')
    path_label.pack()

    # 5. Image Display Area
    image_display_label = tk.Label(root, borderwidth=1, relief="solid")
    image_display_label.pack(pady=10)

    # 6. Result Label
    result_label = tk.Label(root, text="Prediction will appear here.", font=('Arial', 14))
    result_label.pack(pady=10)

    root.mainloop()