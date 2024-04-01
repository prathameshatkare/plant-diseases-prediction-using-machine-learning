import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set the title of the app
st.set_page_config(page_title="Plant Disease Predictor")

# Load the pre-trained model
model = load_model('plant_disease_model.h5')

# Define the class labels
class_labels = ['Healthy', 'Diseased']

# Function to preprocess the input image
def preprocess_image(image_path):
    """
    Preprocess the input image for the plant disease prediction model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Preprocessed image as a numpy array.
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit app
def main():
    """
    Main function for the Streamlit app.
    """
    st.title("Plant Disease Predictor")
    st.write("Upload an image of a plant leaf to predict if it is healthy or diseased.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image_array = preprocess_image(uploaded_file)

        # Make the prediction
        prediction = model.predict(image_array)
        predicted_class = class_labels[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        # Display the prediction
        st.write(f"The plant leaf is predicted to be **{predicted_class}** with a confidence of **{confidence:.2f}**.")

if __name__ == '__main__':
    main()
