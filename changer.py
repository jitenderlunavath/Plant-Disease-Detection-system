from tensorflow.keras.models import load_model

# Load the existing .h5 model
h5_model_path = "D:/study/Projects/plant_disease_model.h5"  # Replace with your actual .h5 model filename
model = load_model(h5_model_path)

# Save the model in the new .keras format
keras_model_path = "D:/study/projects/plant_model.keras"  # Replace with desired output filename
model.save(keras_model_path)

print(f"Model successfully converted: {h5_model_path} → {keras_model_path}")
