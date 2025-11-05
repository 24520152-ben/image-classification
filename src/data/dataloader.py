import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_datasets(train_dir, VAL_SPLIT=0.2, IMG_SIZE=(224, 224), BATCH_SIZE=32, SEED=24520152, SHUFFLE_SIZE=1000):
    """
    Create Tensorflow datasets for training and validation

    Args:
        train_dir (str): Path to the training image directory
        VAL_SPLIT (float): Fraction of data to use for validation
        IMG_SIZE (tuple): Target image size
        BATCH_SIZE (int): Batch size
        SEED (int): Random seed for reproducibility
        SHUFFLE_SIZE (int): Buffer size for shuffling training data

    Returns:
        train_ds, val_ds (tf.data.Dataset): Preprocessed datasets
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds