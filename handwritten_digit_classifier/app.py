# app.py
import streamlit as st
import numpy as np
import cv2
from model import load_trained_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

model = load_trained_model()

st.title("✍️ Handwritten Digit Classifier (MNIST)")
st.markdown("Draw a digit below:")


canvas = st_canvas(
    fill_color="#000000", 
    stroke_width=10,
    stroke_color="#FFFFFF",  
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)


if st.button("Predict"):
    if canvas.image_data is not None:
        
        img = canvas.image_data[:, :, 0]  
        img = cv2.resize(img, (28, 28))
        img = np.invert(img) 
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])
