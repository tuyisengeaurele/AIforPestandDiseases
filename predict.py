import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
MODEL_NAME = 'sericulture_disease_and_pest_detector_model.h5'

CLASS_NAMES = ['bad_silkworms', 'disease_free', 'good_silkworms', 'leaf_rust', 'leaf_spot']

model = None


# --- 2.loading the model
def load_trained_model():
    global model
    try:
        model = load_model(MODEL_NAME, compile=False)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Failed to load model.")
        root.destroy()


# --- 3. image preprocessing
def prepare_image_for_model(img_path):
    try:
        # loading the image and resizing it
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))

        img_array = image.img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error preparing image: {e}")
        return None


# --- 4. prediction logic
def classify_image(file_path):
    if model is None:
        messagebox.showwarning("Warning", "AI model not loaded.")
        return

    # preprocess
    test_image_array = prepare_image_for_model(file_path)
    if test_image_array is None:
        return

    # predict
    predictions = model.predict(test_image_array)

    # interpret results
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    predicted_class = CLASS_NAMES[predicted_class_index]

    # format and display the result
    result_text = (
        f"Prediction: {predicted_class.replace('_', ' ').title()}\n"

    )
    result_label.config(text=result_text, fg="#004d00", font=('Arial', 14, 'bold'))


# --- 5. GUI
def browse_image_file():
    file_path = filedialog.askopenfilename(
        title="Select Sericulture Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # display the image in the GUI
    try:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        image_display_label.config(image=img_tk)
        image_display_label.image = img_tk  # Keep a reference to prevent garbage collection

        path_label.config(text=f"Selected: {file_path.split('/')[-1]}")

        # run the classification
        classify_image(file_path)

    except Exception as e:
        messagebox.showerror("File Error", f"Could not process the selected file: {e}")



if __name__ == '__main__':
    root = tk.Tk()
    root.title("Sericulture AI Predictor")
    root.geometry("400x550")
    root.resizable(False, False)

    # 1. load model
    load_trained_model()

    # 2. title label
    title_label = tk.Label(root, text="Sericulture pests and disease detector", font=('Arial', 16, 'bold'), pady=10)
    title_label.pack()

    # 3. browse button
    browse_button = tk.Button(
        root,
        text="select image",
        command=browse_image_file,
        font=('Arial', 12),
        bg="#4CAF50",
        fg="white",
        pady=5
    )
    browse_button.pack(pady=10)

    # 4. path label
    path_label = tk.Label(root, text="No image selected", font=('Arial', 10), fg='gray')
    path_label.pack()

    # 5. image display area
    image_display_label = tk.Label(root, borderwidth=1, relief="solid")
    image_display_label.pack(pady=10)

    # 6. result Label
    result_label = tk.Label(root, text="Prediction will appear here.", font=('Arial', 14))
    result_label.pack(pady=10)

    root.mainloop()