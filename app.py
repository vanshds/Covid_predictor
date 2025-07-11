import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import os
import requests

# ---------- Background Setup ----------
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            filter: brightness(0.85);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg_image("coronavirus.jpg")  # Make sure this image is in the repo

# ---------- Model Download & Load ----------
MODEL_PATH = "covid_cnn_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=FILE_ID" 

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

@st.cache_resource
def load_cnn_model():
    download_model()
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# ---------- Page Setup ----------
st.set_page_config(page_title="COVID Chest X-ray Classifier", layout="centered")
st.title("🩺 Chest X-ray COVID-19 Detector")
st.write("Upload a chest X-ray image and let the CNN model predict if it's **COVID Positive** or **Normal**.")

# ---------- Upload and Prediction ----------
uploaded = st.file_uploader("Upload a Chest X-ray (PNG/JPG/JPEG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    image_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        label = "Normal"
        st.markdown(f"### ✅ Prediction: **{label}**")
    else:
        label = "COVID Positive"
        st.markdown(f"### ⚠️ Prediction: **{label}**")
        st.error("Please consult a doctor immediately. Follow these precautions:")
        st.markdown("""
        ### 🛡️ Suggested Precautions & Care:
        - 🏠 Isolate yourself to prevent spreading.
        - 🌡️ Monitor oxygen level (use oximeter).
        - 💧 Stay hydrated and rest.
        - 💊 Take prescribed antivirals (only under doctor’s advice).
        - ☎️ Contact healthcare services if symptoms worsen.
        - 😷 Wear a mask even at home.
        """)

# ---------- Footer ----------
st.markdown("---")
st.caption("⚠️ This app is for educational purposes only. Always rely on professional medical advice.")


