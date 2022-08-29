import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('best_model.hdf5')

st.write("""
         # Food item Prediction
         """
         )

st.write("This is a simple image classification web app to predict food item name")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a apple pie!")
    elif np.argmax(prediction) == 1:
        st.write("It is a bibimbap!")
    elif np.argmax(prediction) == 2:
        st.write("It is a cannoli!")
    elif np.argmax(prediction) == 3:
        st.write("It is a edamame!")
    elif np.argmax(prediction) == 4:
        st.write("It is a falafel!")
    elif np.argmax(prediction) == 5:
        st.write("It is a french toast!")
    elif np.argmax(prediction) == 6:
        st.write("It is a ice cream!")
    elif np.argmax(prediction) == 7:
        st.write("It is a ramen!")
    elif np.argmax(prediction) == 8:
        st.write("It is a sushi!")
    else:
        st.write("It is a tiramisu!")
    
    st.text("Probability (0: apple pie, 1: bibimbap, 2: cannoli, 3: edamame, 4: falafel, 5: toast, 6: ice cream, 7: ramen, 8: sushi, 9: tiramisu")
    st.write(prediction)