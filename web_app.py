import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="centered"
)

# Title
st.title("🫁 Lung Cancer Detection System")
st.write("Upload a lung CT scan image and the AI model will predict if cancer is present.")
st.subheader("Model Performance")

st.image("accuracy_graph.png", caption="Model Accuracy Graph")

st.image("confusion_matrix.png", caption="Confusion Matrix")

IMG_SIZE = 128

# Load trained model
model = load_model("model/lung_cancer_model.h5")

# Upload image
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
     img = Image.open(uploaded_file).convert("RGB")
     img = img.resize((128,128))

     img = np.array(img)
     img = img / 255.0
     img = img.reshape(1,128,128,3)

     if st.button("Analyze Image"):
         prediction = model.predict(img)[0][0]

         confidence = round(max(prediction, 1 - prediction) * 100, 2)

         st.write("### Prediction Confidence")
         st.progress(int(confidence))

         if prediction > 0.5:
             st.success(f"✅ Normal Lung (Confidence: {confidence}%)")
         else:
             st.error(f"⚠ Cancer Detected (Confidence: {confidence}%)")

            