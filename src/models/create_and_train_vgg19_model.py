import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_and_train_vgg19_model(train_dataset, validation_dataset, epochs=20):
    """
    Create and train a VGG19-based convolutional neural network model with data augmentation.

    Parameters:
    train_dataset (tf.data.Dataset): The training dataset.
    validation_dataset (tf.data.Dataset): The validation dataset.
    epochs (int): Number of training epochs (default is 20).

    Returns:
    model (tf.keras.Model): The trained VGG19-based model.
    history (tf.keras.callbacks.History): Training history.
    """

    # Base VGG model
    base_vgg_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_vgg_model.trainable = False

    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomBrightness(0.4)
    ])

    # Definition of a CNN model with data augmentation based on VGG architecture
    model = Sequential([
        data_augmentation,
        base_vgg_model,
        Conv2D(512, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model using the Adam optimizer, binary_crossentropy loss function, and accuracy metric.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stopping with patience of 4 epochs
    earlystop = EarlyStopping(patience=4)

    # Define learning rate reduction callback to monitor validation loss, reduce by a factor of 0.2, and set a minimum learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=4,
                                                verbose=2,
                                                factor=0.2,
                                                min_lr=0.0001)

    # Combine both callbacks into a list
    callbacks = [earlystop, learning_rate_reduction]

    # Train the model using the training dataset, validate on the validation dataset,
    # run for the specified number of epochs, display training progress, and apply defined callbacks.
    history = model.fit(train_dataset, validation_data=validation_dataset,
                        epochs=epochs, verbose=1, callbacks=callbacks)

    return model, history