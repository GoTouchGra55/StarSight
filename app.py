import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.title("StarSight")

root = os.getcwd()
model = YOLO(os.path.join(root, "./model/best.pt"))

uploaded_file = st.file_uploader("Upload an image of the night sky", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width="stretch")

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run YOLO inference
    results = model(img)
    annotated = results[0].plot()

    # Show detection
    st.image(annotated, caption="Detected Constellations", channels="BGR")

    # Show detected names
    detections = results[0].boxes
    if detections is not None and len(detections) > 0:
        class_indices = detections.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_indices]
        st.write("## Detected: ", ", ".join(set(class_names)))