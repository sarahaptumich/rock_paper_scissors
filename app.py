import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('rps_model.h5')

# Define class labels
class_names = ['Paper', 'Rock', 'Scissors']  # adjust if your folder labels differ

# UI layout
st.title("🧠 Rock-Paper-Scissors Classifier")
st.write("Upload a photo of a hand showing **rock**, **paper**, or **scissors**, and the model will try to guess it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # shape (1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.markdown(f"### 🧾 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence:.2%}`")