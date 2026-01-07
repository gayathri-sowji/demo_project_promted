import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Face Identification App",
    layout="centered"
)

st.title("🧑 Face Identification using Streamlit")
st.write("Upload an image and the app will detect human faces.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.subheader("📷 Uploaded Image Preview")
    st.image(image, use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Face detection parameters (user control)
    st.subheader("⚙️ Detection Parameters")
    scaleFactor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
    minNeighbors = st.slider("Min Neighbors", 3, 10, 5)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )

    # Draw rectangles and labels
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show result
    st.subheader("✅ Detection Result")
    if len(faces) > 0:
        st.success(f"{len(faces)} face(s) detected")
        st.image(img_array, use_column_width=True)
    else:
        st.warning("No human face detected in the image.")
