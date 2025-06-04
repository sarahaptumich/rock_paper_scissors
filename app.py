import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load your trained model
model = tf.keras.models.load_model("rps_model.h5")
class_names = ['Paper', 'Rock', 'Scissors']

st.title("🧠 Rock-Paper-Scissors Classifier")
st.write("Upload a photo or use your webcam to take a picture.")

# Mode selection
mode = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

elif mode == "Use Webcam":
    picture = st.camera_input("Take a photo")
    if picture:
        image = Image.open(picture).convert('RGB')

# Prediction
if image:
    st.image(image, caption="Input Image", use_container_width=True)
    
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🧾 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence:.2%}`")
