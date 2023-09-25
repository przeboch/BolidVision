import keras
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries


def explain_image(img_path, model, size=(244, 244), top_labels=2, hide_color=1, num_samples=1000):
    """
    Explain an image prediction using LIME (Local Interpretable Model-agnostic Explanations).

    Args:
        img_path (str): The path to the image file to explain.
        model: The machine learning model for prediction.
        size (tuple): The size to which the image should be resized.
        top_labels (int): Number of top labels to explain.
        hide_color (int): The color to use for hiding parts of the image (0 for black).
        num_samples (int): Number of random samples to use for explanation.

    Returns:
        Explanation: An explanation of the image prediction.
    """

    def get_img_array(img_path, size):
        """
        Load and preprocess the image from the given path.

        Args:
            img_path (str): The path to the image file.
            size (tuple): The size to which the image should be resized.

        Returns:
            numpy.ndarray: A processed image array.
        """
        img = keras.utils.load_img(img_path, target_size=size)
        array = keras.utils.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    # Load and preprocess the image
    image_array = get_img_array(img_path, size)

    # Create a LIME Image Explainer
    explainer = LimeImageExplainer()

    # Explain the image prediction
    explanation = explainer.explain_instance(
        image_array[0].astype('double'),
        model.predict,
        top_labels=top_labels,
        hide_color=hide_color,
        num_samples=num_samples,
        batch_size=32
    )

    return explanation


def visualize_explanation(explanation, positive_only=True, num_features=10, hide_rest=False):
    """
    Visualizes an explanation generated by an interpretability model.

    Args:
        explanation (Explanation): An explanation object generated by an interpretability model.
        positive_only (bool, optional): Whether to visualize only positive features. Default is True.
        num_features (int, optional): Number of top features to visualize. Default is 10.
        hide_rest (bool, optional): Whether to hide the rest of the image. Default is False.

    Returns:
        None

    Displays an image with highlighted features based on the explanation.

    """
    # Get the image and mask for the top label in the explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=positive_only,
                                                num_features=num_features, hide_rest=hide_rest)

    # Display the image with marked boundaries
    plt.imshow(mark_boundaries(temp / 255, mask))

    # Add a legend in the upper right corner
    legend_x = temp.shape[1] - 10
    legend_y = 10

    # Set the legend background style
    legend_bbox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'square,pad=0.5'}

    # Text to display in the legend
    legend_text = f'Positive Features: {positive_only}\nNumber of Features: {num_features}'

    # Add legend to the plot
    plt.text(legend_x, legend_y, legend_text, color='black', backgroundcolor='none', fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=legend_bbox)

    # Turn off the axis
    plt.axis("off")
    plt.show()


def visualize_explanation_heatmap(explanation):
    """
    Visualizes a heatmap of local explanations.

    Args:
        explanation: An explanation object containing local explanation information.
    """
    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]

    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap='coolwarm', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.show()

def explain_and_save_images_in_directory(directory_path, model, output_directory):
    """
    Explain and save heatmap visualizations for images in a directory using a model.

    Args:
        directory_path (str): The path to the directory containing the images.
        model: The machine learning model used for image explanations.
        output_directory (str): The directory where the heatmap visualizations will be saved.

    Returns:
        None
    """
    # Create a dictionary to store explanations
    explanations_dict = {}

    # Iterate through the images in the directory
    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)
        explanation = explain_image(img_path, model)
        explanations_dict[filename] = explanation

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through the explanations and save the plots
    for image_name, explanation in explanations_dict.items():
        # Create a subplot with the appropriate number of rows and columns.
        plt.figure()

        # Call the visualize_explanation_heatmap function with the respective data.
        visualize_explanation_heatmap(explanation)

        # Determine the filename for saving this plot in the output directory.
        save_path = os.path.join(output_directory, f"heatmap_{image_name}")

        plt.show()


        # Save the current plot to a file.
        plt.savefig(save_path, dpi = 150)
        plt.close()
