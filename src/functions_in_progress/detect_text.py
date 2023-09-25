import os
from google.cloud import vision

def detect_text(path, credentials_path):
    """
    Detects text in an image using the Google Cloud Vision API.

    Args:
        path (str): The path to the image file to be processed.
        credentials_path (str): The path to the Google Cloud Vision API JSON credentials file.

    Returns:
        None

    Raises:
        Exception: If there is an error in the API response.

    Note:
        This function uses the Google Cloud Vision API to detect text in the provided image.
        It sets the credentials using the `credentials_path`, reads the image file from `path`,
        and prints the detected text along with bounding box information.

        Make sure to install the necessary dependencies and authenticate with the Google Cloud
        service before using this function.
    """

    # Set the Google Cloud Vision API credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    # Create a Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(path, "rb") as image_file:
        content = image_file.read()

    # Create a Vision API image object
    image = vision.Image(content=content)

    # Perform text detection on the image
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    # Print detected text and bounding box information
    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("Bounds: {}".format(",".join(vertices)))

    # Raise an exception if there is an error in the API response
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )