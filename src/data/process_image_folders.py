import os
from PIL import Image
import imghdr

def process_image_folders(main_folder):
    """
    Process image folders in the specified root folder. This function iterates through subfolders
    in the main folder identifies JPEG images, converts them and handles unrecognized image files, deleting them after consent.

    Args:
        main_folder (str): Path to the main folder containing subfolders with images.

    Returns:
        None
    """
    # Loop through folders in 'main_folder'
    for sub_folder in os.listdir(main_folder):
        sub_folder_path = os.path.join(main_folder, sub_folder)

        # Check if the element is a folder
        if os.path.isdir(sub_folder_path):
            # Count JPEG files in the folder
            jpeg_files = [file for file in os.listdir(sub_folder_path) if file.lower().endswith(('.jpg', '.jpeg'))]

            # If there are JPEG files in the folder, process them
            if jpeg_files:
                for file in jpeg_files:
                    try:
                        img = Image.open(os.path.join(sub_folder_path, file))
                        img.save(os.path.join(main_folder + '_jpg', sub_folder, file))
                    except Exception as e:
                        print(f"Error while processing file {file}: {str(e)}")
                        # Ask for confirmation before deleting the file
                        delete_file = input(f"Do you want to delete file {file}? (Y-yes, N-no): ").lower()
                        if delete_file == 'y':
                            os.remove(os.path.join(sub_folder_path, file))
            else:
                print(f"No JPEG files in folder: {sub_folder}")

            # Check the file type for remaining files
            for file in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, file)
                if not os.path.isdir(file_path):
                    file_type = imghdr.what(file_path)
                    if file_type and file_type != 'jpeg':
                        print(f"Unrecognized file format for {file}: {file_type}")
                        # Ask for confirmation before deleting the file
                        delete_file = input(f"Do you want to delete file {file}? (y-yes, n-no): ").lower()
                        if delete_file == 'y':
                            os.remove(file_path)
        else:
            print(f"Element {sub_folder} is not a folder")