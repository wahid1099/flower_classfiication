import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# Optional: Force CPU for debugging (if needed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load your trained Keras model
model = tf.keras.models.load_model('./my_image_classification_model.h5')
class_names = ['Chrysanthemum', 'Hibiscus', 'Marigold', 'Petunia', 'Rose']

# Automatically get expected input size from the model
expected_input_shape = model.input_shape[1:3]  # e.g., (150, 150)

# App Title
st.title("🌸 Flower Classification App")
st.write("Upload a flower image and let the model predict the flower class.")

# File uploader
uploaded_file = st.file_uploader("Choose a flower image...", type=['jpg', 'jpeg', 'png'])

# Prediction function
def predict_image(image):
    # Resize to match model's input shape
    img = cv2.resize(image, expected_input_shape)
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)     # Add batch dimension

    # Predict
    prediction = model.predict(img)

    # Get class with highest confidence
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return class_names[predicted_class], confidence

# Main interface
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to array for OpenCV
    img_array = np.array(image)

    with st.spinner("🔍 Classifying..."):
        class_name, confidence = predict_image(img_array)

    st.write("## Prediction:")
    if confidence < 0.5:
        st.warning("⚠️ No matching flower class found (Low Confidence)")
    else:
        st.success(f"✅ Predicted: **{class_name}** ({confidence * 100:.2f}% confidence)")
