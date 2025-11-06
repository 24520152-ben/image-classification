import tensorflow as tf
from tensorflow.keras import applications, callbacks
from data_loader import create_datasets
from model_builders import MODEL_BUILDERS
import os
import pandas as pd

tf.random.set_seed(24520152)

PREPROCESSOR = {
    "EfficientNetB5": applications.efficientnet.preprocess_input,
    "DenseNet121": applications.densenet.preprocess_input,
    "ResNet50V2": applications.resnet_v2.preprocess_input,
    "MobileNetV2": applications.mobilenet_v2.preprocess_input,
}

def train_and_save_model(model_name, train_dir, save_dir, num_classes, VAL_SPLIT=0.2, IMG_SIZE=(224, 224), BATCH_SIZE=32, DROPOUT_RATE=0.3, EPOCHS=10):
    # Prepare preprocess function and datasets
    preprocess_fn = PREPROCESSOR[model_name] 
    train_ds, val_ds = create_datasets(train_dir=train_dir, preprocess_fn=preprocess_fn, VAL_SPLIT=VAL_SPLIT, IMG_SIZE=IMG_SIZE, BATCH_SIZE=BATCH_SIZE)

    # Build model
    builder = MODEL_BUILDERS[model_name]
    model = builder(INPUT_SHAPE=(*IMG_SIZE, 3), NUM_CLASSES=num_classes, DROPOUT_RATE=DROPOUT_RATE)
    print(model.summary())

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Set up callbacks
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.keras")

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=1,
        )
    ]

    # Train model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callback_list,
        validation_data=val_ds,
    )

    return history

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
    SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
    HIS_DIR = os.path.join(BASE_DIR, "history")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(HIS_DIR, exist_ok=True)

    class_names = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    NUM_CLASSES = len(class_names)

    MODELS = [
        "EfficientNetB5",
        "DenseNet121",
        "ResNet50V2",
        "MobileNetV2",
    ]

    for model_name in MODELS:
        print(f"\nTraining model: {model_name}")
        history = train_and_save_model(
            model_name=model_name,
            train_dir=TRAIN_DIR,
            save_dir=SAVE_DIR,
            num_classes=NUM_CLASSES,
            EPOCHS=5,
        )
        pd.DataFrame(history.history).to_csv(os.path.join(HIS_DIR, f"{model_name}_history.csv"), index=False)