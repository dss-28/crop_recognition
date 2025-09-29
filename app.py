import streamlit as st
import pandas as pd
import cv2
import numpy as np
import joblib
import os
my_dict={0:"jute",1:"maize",2:"rice",3:"sugarcane",4:"wheat"}
# Initialize session state for selected image index
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None

# Feature extraction function
def extract_color_histogram(image):
    if image is None:
        raise ValueError("Invalid image array.")

    image = cv2.resize(image, (224, 224))  # Resize for consistency

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR to HSV since OpenCV loads in BGR
    hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()  # Shape: (1, 512)

# Load scaler and model
scaler = joblib.load("scaler.joblib")
model = joblib.load("model.joblib")

st.title("🌾 Crop Identification App ")

# Load CSV containing image paths
csv_path = "test.csv"  # Update path if needed
df = pd.read_csv(csv_path)

if 'path' not in df.columns:
    st.error("CSV must have a 'path' column with image file paths.")
    st.stop()

# Filter valid image paths and load images resized
valid_paths = []
image_list = []

for p in df['path']:
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            image_list.append(img)
            valid_paths.append(p)

if not image_list:
    st.error("No valid images found!")
    st.stop()

# Show prediction and selected image at the top
if st.session_state.selected_index is not None:
    selected_image = image_list[st.session_state.selected_index]
    selected_path = valid_paths[st.session_state.selected_index]

    st.subheader("Selected Image")
    st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), caption=f"Selected: {os.path.basename(selected_path)}", use_container_width=False,width=400)

    features = extract_color_histogram(selected_image).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    st.success(f"✅ Predicted Crop: **{my_dict[prediction]}**")

# Show images in rows of 4 columns max
images_per_row = 4

for i in range(0, len(image_list), images_per_row):
    row_images = image_list[i:i+images_per_row]
    row_paths = valid_paths[i:i+images_per_row]
    cols = st.columns(len(row_images))
    for j, (col, img, path) in enumerate(zip(cols, row_images, row_paths)):
        with col:
            if st.button(f"Select {os.path.basename(path)}", key=path):
                st.session_state.selected_index = i + j
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=os.path.basename(path), use_container_width=True)
