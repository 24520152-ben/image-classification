import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import applications
import keras_hub

def build_efficientnet(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an EfficientNetB0 model with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="EfficientNetB0_FeatureExtractor")
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
        input_tensor=input_layer,
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="DenseNet121_FeatureExtractor")
    return model

def build_visiontransformer(INPUT_SHAPE=(224, 224, 3), NUM_CLASSES=6, DROPOUT_RATE=0.3):
    """
    Build an Vision Transformer model with a custom head for classification
    This uses feature extraction (freezing the backbone)

    Args:
        INPUT_SHAPE (tuple): The shape of the input images
        NUM_CLASSES (int): The number of output classes
        DROPOUT_RATE (float): Dropout rate for the classifier head

    Returns:
        tf.keras.Model: The compiled Keras model
    """
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")

    backbone = keras_hub.models.Backbone.from_preset(
        "vit_base_patch16_224_imagenet"
    )
    backbone.trainable = False

    x = backbone(input_layer)
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="VisionTransformer_FeatureExtractor")
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
        input_tensor=input_layer,
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet50V2_FeatureExtractor")
    return model