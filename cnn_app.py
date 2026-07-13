%%writefile cnn_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


st.title("Digit Recognition using CNN (MNIST)")

st.write(
    "Upload an image of a handwritten digit"
)


# Load Model

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "mnist_cnn.h5"
    )

    return model


model = load_model()



# Upload Image

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg","jpeg","png"]
)



if uploaded_file is not None:

    # Read Image

    image = Image.open(
        uploaded_file
    ).convert("L")


    # Convert to MNIST style

    image = ImageOps.invert(image)

    image = image.resize(
        (28,28)
    )


    # Display image

    st.image(
        image,
        caption="Uploaded Digit",
        width=150
    )


    # Preprocess

    img_array = np.array(image)

    img_array = img_array.reshape(
        1,
        28,
        28,
        1
    )

    img_array = img_array / 255.0



    # Prediction

    prediction = model.predict(
        img_array
    )


    digit = np.argmax(
        prediction
    )


    confidence = np.max(
        prediction
    ) * 100



    st.success(
        f"Predicted Digit: {digit}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )
