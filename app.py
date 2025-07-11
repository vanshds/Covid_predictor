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
            filter: brightness(0.9);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg_image("coronavirus.jpg")

# ---------- Constants ----------
MODEL_ID = "4T1R6STQwfZeDgkAloZ0RM3lYRKgj69p"  # Replace with your actual file ID
MODEL_PATH = "covid_cnn_model.h5"

# ---------- Download model from Google Drive ----------
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# ---------- Load Model with caching ----------
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            download_file_from_google_drive(MODEL_ID, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# ---------- Streamlit page config ----------
st.set_page_config(page_title="COVID Chest X-ray Classifier", layout="centered")
st.title("🩺 Chest X-ray COVID-19 Detector")
st.write("Upload a chest X-ray image and let the CNN model predict if it's **COVID Positive** or **Normal**.")

# ---------- Upload & Predict ----------
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

# ---------- Footer ----------
st.markdown("---")
st.caption("⚠️ This app is for educational purposes only. Always rely on professional medical advice.")


