import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import PIL.Image
import os


def display_images_from_dataset(dataset, num_images=6):
    """
    Displays a specified number of random images from a given dataset along with their class names.

    Args:
        dataset (tf.data.Dataset): The dataset to display images from.
        num_images (int, optional): The number of random images to display. Default is 6.
    """
    # Get class names from the dataset
    class_names = dataset.class_names

    # Get the first batch of data from the dataset
    for images, labels in dataset.take(1):
        # Choose num_images random indices of images
        num_total_images = images.shape[0]
        random_indices = np.random.choice(num_total_images, num_images, replace=False)

        # Select the images and labels corresponding to the random indices
        selected_images = tf.gather(images, random_indices)
        selected_labels = tf.gather(labels, random_indices)

        # Display the selected images with their class names
        plt.figure(figsize=(12, 8))
        for i in range(num_images):
            plt.subplot(2, 3, i + 1)
            plt.imshow(selected_images[i].numpy().astype("uint8"))
            label_index = int(selected_labels[i].numpy())
            plt.title(f"Class: {class_names[label_index]}")
            plt.axis("off")

        plt.show()

def plot_history(history):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history (keras.callbacks.History): The training history obtained from a Keras model.fit() call.

    Returns:
        None
    """

    # Plot the Loss Curves
    plt.figure(figsize=[6, 4])
    plt.plot(history.history['loss'], 'o--', linewidth=3.0, label='Training loss')
    plt.plot(history.history['val_loss'], 'p--', linewidth=3.0, label='Validation Loss')
    plt.legend(fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Plot the Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'o--', linewidth=3.0, label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'p--', linewidth=3.0, label='Validation Accuracy')
    plt.legend(fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

def plot_confusion_matrix(true_labels, predicted_labels, class_labels=None):
    """
    Plot a confusion matrix using true labels and predicted labels.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.
        class_labels (list, optional): List of class labels. Default is None.

    Returns:
        None
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


def display_images_from_folder(folder, num_cols=3, num_to_display=None, show_titles=True):
    """
    Displays images from a folder on a dynamically adjusted layout.

    :param folder: Path to the folder containing the images.
    :param num_cols: Number of columns for the image display grid.
    :param num_to_display: Number of images to display. If None, all images in the folder will be displayed.
    :param show_titles: Whether to display titles for the images.
    """
    image_list = os.listdir(folder)

    if num_to_display is not None:
        image_list = image_list[:num_to_display]

    num_images = len(image_list)

    if num_images == 0:
        print("No images found in the folder.")
        return

    # Calculate the number of rows based on the number of columns
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create subplots
    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for i, image_file in enumerate(image_list):
        image_path = os.path.join(folder, image_file)
        image = PIL.Image.open(image_path)

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')

        if show_titles:
            plt.title(image_file)

    plt.tight_layout()
    plt.show()