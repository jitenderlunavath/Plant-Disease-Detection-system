import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from tkinter import Tk, filedialog, Label, Button, Text, Scrollbar
from PIL import Image, ImageTk

# Define dataset paths
DATASET_DIR = "D:/study/Projects/disease_detection/plant_dataset"  # Update with the correct dataset folder
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
MODEL_PATH = "plant_disease_model.keras"  # Use .keras format

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Extract class labels
label_binarizer_classes = list(train_generator.class_indices.keys())

# Check if a trained model exists
if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    # Build CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_binarizer_classes), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, validation_data=valid_generator, epochs=10)

    # Save the model in .keras format
    model.save(MODEL_PATH)
    print("Model trained and saved successfully!")

# GUI Section
def predict_disease(image_path, model, label_binarizer_classes):
    image = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = tf.keras.utils.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = label_binarizer_classes[np.argmax(prediction)]
    print(f"Predicted Disease: {predicted_label}")  # Debugging Output
    return predicted_label

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        result = predict_disease(file_path, model, label_binarizer_classes)
        result_text.config(state='normal')
        result_text.delete(1.0, "end")
        result_text.insert("end", f"Prediction: {result}")
        result_text.config(state='disabled')

# Create GUI Application
app = Tk()
app.title("Plant Disease Detection")
app.geometry("600x500")
app.configure(bg="#f0f0f0")

# Add Label to display the selected image
panel = Label(app, bg="#dddddd", relief="solid")
panel.pack(pady=10)

# Add Button to upload an image
upload_button = Button(app, text="Upload Image", command=open_file, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_button.pack(pady=10)

# Add Text Widget for result (Copyable Text)
result_text = Text(app, height=2, width=50, font=("Arial", 14), wrap="word", state='disabled', bg="white", relief="solid")
result_text.pack(pady=10)

# Run the application
app.mainloop()
