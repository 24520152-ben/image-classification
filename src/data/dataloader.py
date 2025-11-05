import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_data_augmentation():
    """Create a simple data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

def create_datasets(train_dir, preprocess_fn, VAL_SPLIT=0.2, IMG_SIZE=(224, 224), BATCH_SIZE=32, SEED=24520152):
    """
    Create Tensorflow datasets for training and validation

    Args:
        train_dir (str): Path to the training image directory
        preprocess_fn (callable): The preprocessing function specific to the model (e.g., tf.keras.applications.efficientnet.preprocess_input)
        VAL_SPLIT (float): Fraction of data to use for validation
        IMG_SIZE (tuple): Target image size
        BATCH_SIZE (int): Batch size
        SEED (int): Random seed for reproducibility

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

    AUTOTUNE = tf.data.AUTOTUNE
    data_augmentation = build_data_augmentation()

    def augment_and_preprocess_train(image, label):
        """Apply augmentation and preprocessing to the training set"""
        image = data_augmentation(image, training=True)
        image = preprocess_fn(image)
        return image, label

    def preprocess_val(image, label):
        """Apply preprocessing to the validation set (no augmentation)"""
        image = preprocess_fn(image)
        return image, label

    train_ds = train_ds.cache().map(augment_and_preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().map(preprocess_val, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds