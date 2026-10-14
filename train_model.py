"""Train a deep learning waste classification model with transfer learning."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_PATH = ROOT_DIR / "waste_model.h5"
BEST_MODEL_PATH = ROOT_DIR / "best_waste_model.h5"
CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
HISTORY_PLOT_PATH = ROOT_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = ROOT_DIR / "confusion_matrix.png"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


DEFAULT_CLASSES = ["cardboard", "glass", "metal", "organic", "paper", "plastic", "trash"]


def generate_demo_dataset(dataset_dir: Path, class_names: list[str] | None = None, samples_per_class: int = 20) -> None:
    """Create a tiny synthetic dataset so the project can build and run immediately."""
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = DEFAULT_CLASSES

    palette = {
        "cardboard": (198, 166, 109),
        "glass": (123, 191, 234),
        "metal": (164, 170, 184),
        "paper": (245, 229, 162),
        "plastic": (255, 143, 94),
        "organic": (123, 196, 110),
        "trash": (109, 109, 109),
    }

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(samples_per_class):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            base = np.array(palette.get(class_name, (200, 200, 200)), dtype=np.uint8)
            image[:] = base

            for x_step in range(0, 224, 32):
                for y_step in range(0, 224, 32):
                    patch = np.random.randint(0, 25, size=(16, 16, 3), dtype=np.uint8)
                    image[y_step:y_step + 16, x_step:x_step + 16] = np.clip(
                        base + patch - 10, 0, 255
                    )

            if class_name in {"glass", "plastic"}:
                for radius in range(20, 120, 20):
                    img_draw = Image.fromarray(image)
                    draw = ImageDraw.Draw(img_draw)
                    draw.ellipse(
                        (112 - radius, 112 - radius, 112 + radius, 112 + radius),
                        outline=(255, 255, 255),
                        width=3,
                    )
                    image = np.array(img_draw)

            if class_name in {"metal", "cardboard"}:
                img_draw = Image.fromarray(image)
                draw = ImageDraw.Draw(img_draw)
                for line in range(0, 220, 22):
                    draw.line((line, 0, line + 30, 224), fill=(20, 20, 20), width=2)
                image = np.array(img_draw)

            if class_name == "organic":
                img_draw = Image.fromarray(image)
                draw = ImageDraw.Draw(img_draw)
                for dot in range(30):
                    x = np.random.randint(0, 224)
                    y = np.random.randint(0, 224)
                    draw.ellipse((x, y, x + 8, y + 8), fill=(60, 150, 60))
                image = np.array(img_draw)

            if class_name == "trash":
                img_draw = Image.fromarray(image)
                draw = ImageDraw.Draw(img_draw)
                for stripe in range(10):
                    x0 = np.random.randint(0, 224)
                    y0 = np.random.randint(0, 224)
                    x1 = np.random.randint(0, 224)
                    y1 = np.random.randint(0, 224)
                    draw.line((x0, y0, x1, y1), fill=(30, 30, 30), width=3)
                image = np.array(img_draw)

            image_path = class_dir / f"{index:03d}.png"
            Image.fromarray(image).save(image_path)


def ensure_dataset_exists(dataset_dir: Path) -> list[str]:
    """Validate that the expected dataset structure and classes exist."""
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted(
        [item.name for item in dataset_dir.iterdir() if item.is_dir() and item.name not in {"__pycache__"}]
    )

    if not class_names:
        class_names = DEFAULT_CLASSES
        generate_demo_dataset(dataset_dir, class_names)
        return class_names

    for class_name in class_names:
        image_files = list((dataset_dir / class_name).glob("*.*"))
        if not image_files:
            generate_demo_dataset(dataset_dir, class_names)
            return class_names

    return class_names


def build_augmentation_layer() -> tf.keras.Sequential:
    """Create an image augmentation pipeline for training."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def load_datasets(dataset_dir: Path, class_names: list[str]):
    """Load train, validation, and test sets using tf.data API."""
    train_ds = image_dataset_from_directory(
        directory=str(dataset_dir),
        labels="inferred",
        class_names=class_names,
        validation_split=0.30,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        interpolation="bilinear",
        shuffle=True,
    )

    val_test_ds = image_dataset_from_directory(
        directory=str(dataset_dir),
        labels="inferred",
        class_names=class_names,
        validation_split=0.30,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=None,
        color_mode="rgb",
        interpolation="bilinear",
        shuffle=True,
    )

    total_val_test = tf.data.experimental.cardinality(val_test_ds).numpy()
    if total_val_test <= 1:
        val_ds = val_test_ds.batch(BATCH_SIZE)
        test_ds = val_test_ds.batch(BATCH_SIZE)
    else:
        split_idx = total_val_test // 2
        val_ds = val_test_ds.take(split_idx).batch(BATCH_SIZE)
        test_ds = val_test_ds.skip(split_idx).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


def prepare_datasets(train_ds, val_ds, test_ds):
    """Apply augmentation and caching to the datasets."""
    augmentation = build_augmentation_layer()

    def augment_images(images, labels):
        return augmentation(images, training=True), labels

    train_ds = train_ds.map(augment_images, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def build_model(num_classes: int) -> tf.keras.Model:
    """Create a MobileNetV2 transfer-learning model for classification."""
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_layer")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs, outputs, name="waste_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model: tf.keras.Model, train_ds, val_ds, initial_epoch: int = 0) -> tf.keras.callbacks.History:
    """Unfreeze the final MobileNetV2 layers and continue training with a lower learning rate."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 base model layer not found in model.")

    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(filepath=str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max"),
    ]

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )


def save_training_history(history, file_path: Path):
    """Plot accuracy and loss curves for training and validation."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    train_acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    axs[0].plot(train_acc, label="Train Accuracy")
    axs[0].plot(val_acc, label="Validation Accuracy")
    axs[0].set_title("Accuracy Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_loss, label="Train Loss")
    axs[1].plot(val_loss, label="Validation Loss")
    axs[1].set_title("Loss Curves")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def evaluate_dataset(model: tf.keras.Model, dataset) -> tuple[list[int], list[int]]:
    """Collect ground truth labels and predictions from a dataset."""
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())

    return y_true, y_pred


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], file_path: Path):
    """Generate a confusion matrix and save it as an image."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the full training workflow."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    class_names = ensure_dataset_exists(DATASET_DIR)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    train_ds, val_ds, test_ds = load_datasets(DATASET_DIR, class_names)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    model = build_model(len(class_names))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks,
        verbose=1,
    )

    fine_tune_history = fine_tune_model(model, train_ds, val_ds, initial_epoch=len(history.epoch))

    combined_history = {
        key: history.history.get(key, []) + fine_tune_history.history.get(key, [])
        for key in set(history.history) | set(fine_tune_history.history)
    }

    history.history = combined_history
    save_training_history(history, HISTORY_PLOT_PATH)

    y_true, y_pred = evaluate_dataset(model, test_ds)
    save_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)

    model.save(str(MODEL_PATH), include_optimizer=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:\n")
    print(report)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved class names: {CLASS_NAMES_PATH}")
    print(f"Saved training curves: {HISTORY_PLOT_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
