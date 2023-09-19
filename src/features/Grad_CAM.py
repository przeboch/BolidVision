def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given input image.

    Args:
        img_array (numpy.ndarray): The input image as a NumPy array.
        model (tensorflow.keras.Model): The trained model for which the Grad-CAM heatmap is generated.
        last_conv_layer_name (str): The name of the last convolutional layer in the model.
        pred_index (int, optional): The index of the class for which the heatmap should be generated.
                                    If not provided, the class with the highest predicted probability will be used.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap as a NumPy array.

    This function generates a Grad-CAM (Gradient-weighted Class Activation Map) heatmap for an input image using a
    pre-trained model. It highlights the regions in the input image that are most relevant to the predicted class.

    The algorithm involves the following steps:
    1. Create a model that maps the input image to the activations of the last convolutional layer as well as the output predictions.
    2. Compute the gradient of the top predicted class (or the specified class) with respect to the activations of the last convolutional layer.
    3. Calculate the mean intensity of the gradient over each feature map channel.
    4. Multiply each channel in the feature map array by its importance with regard to the top predicted class.
    5. Sum all the channels to obtain the heatmap class activation.
    6. Normalize the heatmap between 0 and 1 for visualization.

    Example:
    ```
    # Load your model and input image
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    img_array = preprocess_input(load_image('your_image.jpg'))

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3')

    # Visualize the heatmap on the original image
    superimposed_img = superimpose_heatmap(img_array, heatmap)

    # Display or save the superimposed image
    plt.imshow(superimposed_img)
    plt.show()
    ```

    For detailed usage and visualization, you can use the `superimpose_heatmap` function to overlay the heatmap on the original image.
    """
    # Create a model that maps the input image to the activations of the last convolutional layer and predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image with respect to the last convolutional layer activations
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradient of the output neuron (top predicted or chosen) with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Calculate the mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its importance and sum to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_and_display_heatmap(img_path, model, last_conv_layer_name, img_size=(224, 224)):
    """
    Generate and display the class activation heatmap for an image.

    Args:
        img_path (str): Path to the image for which the heatmap should be generated.
        model (keras.Model): The neural network model.
        last_conv_layer_name (str): Name of the last convolutional layer in the model.
        img_size (tuple, optional): Size to which the original image should be resized. Defaults to (224, 224).

    Returns:
        heatmap (numpy.ndarray): The class activation heatmap.
    """
    preprocess_input = keras.applications.vgg16.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    return heatmap


def save_and_display_gradcam(img_path, model, heatmap, last_conv_layer_name, cam_path="cam.jpg", alpha=0.4,
                             img_size=(224, 224)):
    """
    Generate and save a Grad CAM (Gradient-weighted Class Activation Mapping) visualization for an image and display it.

    Args:
        img_path (str): Path to the original image.
        model (keras.Model): The neural network model to be used for Grad CAM visualization.
        heatmap: Heatmap generated using Grad CAM.
        last_conv_layer_name (str): Name of the last convolutional layer in the model.
        cam_path (str, optional): Path to save the Grad CAM image. Defaults to "cam.jpg".
        alpha (float, optional): Blending factor for superimposing the heatmap on the original image. Defaults to 0.4.
        img_size (tuple, optional): Size to which the original image should be resized. Defaults to (224, 224).

    Returns:
        None
    """
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Heatmap
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    model.layers[-1].activation = None
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use the "plasma" colormap to colorize heatmap
    jet = mpl_cm.get_cmap("plasma")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))