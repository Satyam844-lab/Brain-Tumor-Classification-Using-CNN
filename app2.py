
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)


# load model
model = tf.keras.models.load_model("brain_tumor_4class_model.keras")


class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


# title
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to classify tumor type using Deep Learning.")


uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])


if uploaded_file:

    col1, col2 = st.columns(2)


    image = Image.open(uploaded_file)

    img = np.array(image)


    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = img.reshape(1,128,128,3)


    with st.spinner("Analyzing MRI..."):

        prediction = model.predict(img)


    class_index = np.argmax(prediction)

    confidence = prediction[0][class_index]


    # left side image
    with col1:
        st.image(image, caption="Uploaded MRI", use_column_width=True)


    # right side result
    with col2:

        st.subheader("Prediction")

        st.success(class_names[class_index])

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )


        st.subheader("Class Probabilities")

        fig = plt.figure()

        plt.bar(class_names, prediction[0])

        plt.xticks(rotation=30)

        plt.ylabel("Probability")

        st.pyplot(fig)
