import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# A dictionary to map the model's output to human-readable class names
class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# Use Streamlit's cache to load the model only once. This is crucial for performance.
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model from the HDF5 file."""
    try:
        model = keras.models.load_model('fashion_mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Set up the Streamlit app interface
st.title("Fashion MNIST Classifier")
st.write("Upload an image of clothing to get a prediction!")
st.info("The model was trained on 28x28 pixel grayscale images from the Fashion MNIST dataset. For best results, please upload a similar type of image.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Check if the model loaded successfully
    if model:
        # Preprocess the image for the model
        st.spinner("Predicting...")
        
        # Resize the image to 28x28 pixels
        image = image.resize((28, 28))
        
        # Convert the image to a NumPy array
        img_array = np.asarray(image)
        
        # Normalize the pixel values (same as in train.py)
        img_array = img_array / 255.0
        
        # Expand dimensions to create a batch of 1
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        # Display the prediction result
        st.success(f"Prediction: **{predicted_class_name}**")
        
        # Optionally display the confidence score
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
