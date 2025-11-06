import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import applications
import keras_hub

def build_efficientnet(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an EfficientNetB5 model based on Swin Transformer train / pretrain recipe with modifications with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = applications.EfficientNetB5(
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(input_layer)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_layer")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="EfficientNetB5")
    return model

def build_densenet(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an DenseNet121 model with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = applications.DenseNet121(
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(input_layer)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_layer")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="DenseNet121")
    return model

def build_resnet(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an ResNet50V2 model with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(input_layer)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_layer")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet50V2")
    return model

def build_mobilenet(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an MobileNetV2 model with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = backbone(input_layer)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_layer")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="MobileNetV2")
    return model

MODEL_BUILDERS = {
    "EfficientNetB5": build_efficientnet,
    "DenseNet121": build_densenet,
    "ResNet50V2": build_resnet,
    "MobileNetV2": build_mobilenet,
}