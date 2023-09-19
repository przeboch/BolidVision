import tensorflow as tf


def create_datasets(train_path, test_path, batch_size=32, image_size=(224, 224), seed=42):
    """
    Create training, validation, and test datasets from image directories.

    Args:
        train_path (str): Path to the training dataset directory.
        test_path (str): Path to the test dataset directory.
        batch_size (int): Batch size for the datasets (default is 32).
        image_size (tuple): Image dimensions (height, width) (default is (224, 224)).
        seed (int): Seed for shuffling (default is 42).

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, and test dataset.
    """

    # Create training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        label_mode='binary',
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset="training",
        seed=seed
    )

    # Create validation dataset
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        label_mode='binary',
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset="validation",
        seed=seed
    )

    # Create test dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        label_mode='binary',
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size
    )

    return train_dataset, validation_dataset, test_dataset