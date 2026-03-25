import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# load trained model
model = tf.keras.models.load_model("brain_tumor_4class_model.keras")


# class labels (same order as training folders)
class_names = [
    "glioma tumor",
    "meningioma tumor",
    "no tumor",
    "pituitary tumor"
]


# title
st.title("Brain Tumor Classification (4 classes)")
st.write("Upload an MRI image to detect tumor type")


# upload image
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg","png","jpeg"])


if uploaded_file is not None:

    # display image
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI", use_column_width=True)


    # convert to array
    img = np.array(image)


    # convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    # resize
    img = cv2.resize(img, (128,128))


    # normalize
    img = img / 255.0


    # reshape
    img = img.reshape(1,128,128,3)


    # prediction
    prediction = model.predict(img)


    # get class index
    class_index = np.argmax(prediction)


    confidence = prediction[0][class_index]


    # result
    st.subheader("Prediction")

    st.success(class_names[class_index])

    st.write("Confidence:", float(confidence))
