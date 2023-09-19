import numpy as np
import os
import requests
from IPython.display import Image, display
from tensorflow import keras
import keras.utils

def display_and_save_img(image_url):
    """
    Downloads and displays an image from a given URL.

    Args:
        image_url (str): The URL of the image to be displayed and saved.

    Returns:
        str: The local path where the image is saved.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error while downloading the image.
        Exception: If any other error occurs during the process.
    """
    try:
        # Extract the filename from the URL
        img_name = os.path.basename(image_url)

        # Get the local path where the image will be saved
        img_path = keras.utils.get_file(img_name, image_url)

        # Display the downloaded image
        display(Image(img_path))

        # Return the local path of the saved image
        return img_path
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_img_array(img_path, size):
    """
    Load and preprocess an image from a given path.

    Args:
        img_path (str): The file path to the image.
        size (tuple): A tuple specifying the target size (height, width) to which the image should be resized.

    Returns:
        numpy.ndarray: A numpy array containing the preprocessed image.
    """
    # Load the image from the specified path and resize it to the target size.
    img = keras.utils.load_img(img_path, target_size=size)

    # Convert the image to a numpy array.
    array = keras.utils.img_to_array(img)

    # Expand the dimensions of the array to make it suitable for model input.
    array = np.expand_dims(array, axis=0)

    return array


def predict(img_path, model, img_size=(224, 224)):
    """
    Predicts the class label and maximum probability score for an input image using a pre-trained model.

    Args:
        img_path (str): The file path to the input image.
        model (keras.Model): The pre-trained neural network model for image classification.
        img_size (tuple): The target size for resizing the input image (default is (224, 224)).

    Returns:
        None: Prints the predicted class label and maximum probability score.
    """
    # Preprocess the input image
    preprocess_input = keras.applications.vgg16.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, img_size))

    # Make predictions using the model
    preds = model.predict(img_array)

    # Find the predicted class label
    predicted_class = np.argmax(preds, axis=1)

    # Assuming 'train_dataset' is defined elsewhere, get class names
    class_names = train_dataset.class_names

    # Get the predicted label and probability score for the top prediction
    predicted_label = class_names[predicted_class[0]]
    probability_score = preds[0][predicted_class[0]]

    # Print the predicted class label and maximum probability score
    print(f"Predicted Class: {predicted_label}")
    print(f"Probability Score: {probability_score:.4f}")

def display_and_predict_img(image_url, model, train_dataset, img_size=(224, 224)):
    """
    Display an image from a given URL, make predictions using a pre-trained model,
    and print the predicted class label and probability score.

    Args:
        image_url (str): URL of the image to be displayed and predicted.
        model (keras.Model): A pre-trained deep learning model for image classification.
        train_dataset: The dataset containing class names for label mapping.
        img_size (tuple): A tuple specifying the target image size (width, height).

    Raises:
        requests.exceptions.HTTPError: If there's an HTTP error while fetching the image.
        Exception: If an error occurs during the prediction process.

    Returns:
        None
    """
    try:
        # Extract the image name from the URL and create a local path to download it
        img_name = os.path.basename(image_url)
        img_path = keras.utils.get_file(img_name, image_url)

        # Display the downloaded image
        display(Image(img_path))

        # Preprocess the image for model input
        preprocess_input = keras.applications.vgg16.preprocess_input
        img_array = preprocess_input(get_img_array(img_path, img_size))

        # Make predictions using the provided model
        preds = model.predict(img_array)

        # Find the predicted class index and class name
        predicted_class = np.argmax(preds, axis=1)
        class_names = train_dataset.class_names
        predicted_label = class_names[predicted_class[0]]

        # Get the probability score for the predicted class
        probability_score = preds[0][predicted_class[0]]

        # Print the prediction results
        print(f"Predicted Label: {predicted_label}")
        print(f"Probability Score: {probability_score:.4f}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")