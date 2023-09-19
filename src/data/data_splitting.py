def create_image_datasets(img_path, test_dataset_path, batch_size=32, image_size=(224, 224), seed=42):
    """
    Create TensorFlow datasets for training, validation, and testing from image directories.

    Args:
        img_path (str): The path to the main image directory containing subdirectories for classes.
        test_dataset_path (str): The path to the test dataset directory.
        batch_size (int): Batch size for training and validation datasets.
        image_size (tuple): Tuple specifying the target image size (height, width).
        seed (int): Seed for shuffling the datasets.

    Returns:
        train_dataset (tf.data.Dataset): The training dataset.
        validation_dataset (tf.data.Dataset): The validation dataset.
        test_dataset (tf.data.Dataset): The test dataset.
    """
    # Create training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_path,
        label_mode='binary',  # Assumes binary classification, change to 'categorical' for multi-class
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,  # 20% of the data will be used for validation
        subset="training",  # This dataset is for training
        seed=seed
    )

    # Create validation dataset
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_path,
        label_mode='binary',  # Assumes binary classification, change to 'categorical' for multi-class
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,  # 20% of the data will be used for validation
        subset="validation",  # This dataset is for validation
        seed=seed
    )

    # Create test dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dataset_path,
        label_mode='binary',  # Assumes binary classification, change to 'categorical' for multi-class
        shuffle=False,  # No need to shuffle the test dataset
        batch_size=batch_size,
        image_size=image_size
    )
    class_names = train_dataset.class_names  # Get the class names from the training dataset

    print("Labels:", class_names)

    return train_dataset, validation_dataset, test_dataset