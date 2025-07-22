import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# Load your trained model
model = tf.keras.models.load_model('./Flower_Recog_Model.h5')
class_names = ['Chrysanthemum', 'Hibiscus', 'Marigold', 'Petunia', 'Rose']

st.title("🌸 Flower Classification App")
st.write("Upload a flower image and let the model predict the class.")

uploaded_file = st.file_uploader("Choose a flower image...", type=['jpg', 'jpeg', 'png'])

def predict_image(image):
    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    class_name, confidence = predict_image(img_array)

    st.write("## Prediction:")
    if confidence < 0.5:
        st.warning("⚠️ No matching flower class found (Low Confidence)")
    else:
        st.success(f"✅ Predicted: **{class_name}** ({confidence:.2%} confidence)")
