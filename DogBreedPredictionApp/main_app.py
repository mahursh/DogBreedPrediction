import numpy as np
import streamlit as st
import cv2
from keras._tf_keras.keras.models import load_model

# Loading model
model = load_model('./deg_breed.h5')

# Names of classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']


# Setting title of app
st.title('Dog Breed Prediction App')
st.markdown('Upload an image of the dog')

# Uploading the dog image
dog_image = st.file_uploader('Choose an image' , type=['png'])
submit = st.button('Predict')

# On predict button click
if submit:
    if dog_image is not None:

        # Convert file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (224, 224))
        # Convert image to 4 dimension
        opencv_image.shape = (1, 224, 224, 3)
        # Make prediction
        Y_pred = model.predict(opencv_image)

        st.title('The Dog Breed Is :  '+ CLASS_NAMES[np.argmax(Y_pred)])