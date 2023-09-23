from rembg import remove
import os
from PIL import Image
from tqdm import tqdm


def background_removal(images_with_bg_path, images_without_bg_path):
    """
    Process images in 'images_with_bg_path', removing their background and saving the results
    in 'images_without_bg_path'.

    Parameters:
        images_with_bg_path (str): The path to the directory containing images with backgrounds.
        images_without_bg_path (str): The path to the directory where processed images without backgrounds will be saved.

    This function iterates through all the image files in 'images_with_bg_path', checks if they have
    '.jpg' or '.png' extensions, and attempts to remove the background using the 'remove' function from the 'rembg' module.
    The resulting images are converted to the RGB format and saved in the 'images_without_bg_path' directory.

    If any errors occur during processing, such as file not found or image processing issues, they are caught,
    and error messages are printed, allowing the function to continue processing other images.

    If any other unexpected exceptions occur while creating directories or processing images,
    a generic error message is printed.

    Note: The 'remove' function used for background removal is from the 'rembg' module.

    Returns:
        None
    """
    try:
        os.makedirs(images_without_bg_path, exist_ok=True)  # Create the destination directory if it doesn't exist

        # Get a list of image files in the source directory
        image_files = [filename for filename in os.listdir(images_with_bg_path) if filename.endswith('.jpg') or filename.endswith('.png')]

        # Use tqdm to monitor progress
        for filename in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(images_with_bg_path, filename)
            output_path = os.path.join(images_without_bg_path, filename)

            try:
                input_image = Image.open(input_path)
                output_image = remove(input_image)
                output_image = output_image.convert("RGB")
                output_image.save(output_path)
            except Exception as e:
                print(f"Failed to process the file {input_path}: {str(e)}")

        print("Finished removing backgrounds from images.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")