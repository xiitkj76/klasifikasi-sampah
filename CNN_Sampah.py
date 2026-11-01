from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasett_preprocessed"
OUTPUT_DIR = BASE_DIR / "cnn_training_output"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
L2_REGULARIZATION = 1e-4


def collect_splits(
    dataset_dir: Path,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    """Collect image paths and make reproducible, per-class data splits."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    class_dirs = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas di: {dataset_dir}")

    train_samples: list[tuple[str, int]] = []
    validation_samples: list[tuple[str, int]] = []
    test_samples: list[tuple[str, int]] = []
    class_names: list[str] = []

    for class_index, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        image_paths = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), str(path).casefold()),
        )
        if len(image_paths) < 3:
            raise ValueError(
                f"Kelas '{class_dir.name}' membutuhkan minimal 3 gambar "
                "agar train, validation, dan test masing-masing terisi."
            )

        random.Random(SEED + class_index).shuffle(image_paths)
        train_count = int(len(image_paths) * TRAIN_RATIO)
        validation_count = int(len(image_paths) * VALIDATION_RATIO)
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
        if train_count + validation_count >= len(image_paths):
            train_count = len(image_paths) - 2
            validation_count = 1

        labeled_paths = [(str(path), class_index) for path in image_paths]
        train_samples.extend(labeled_paths[:train_count])
        validation_samples.extend(
            labeled_paths[train_count : train_count + validation_count]
        )
        test_samples.extend(labeled_paths[train_count + validation_count :])

    return train_samples, validation_samples, test_samples, class_names


def _load_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    samples: list[tuple[str, int]],
    training: bool = False,
) -> tf.data.Dataset:
    if not samples:
        raise ValueError("Tidak dapat membuat dataset dari daftar gambar kosong.")

    paths, labels = zip(*samples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(samples),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=SEED),
                tf.keras.layers.RandomRotation(0.08, seed=SEED + 1),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.1, 0.1),
                    width_factor=(-0.1, 0.1),
                    seed=SEED + 2,
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=0.08,
                    width_factor=0.08,
                    seed=SEED + 3,
                ),
            ],
            name="geometric_augmentation",
        )

        def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            image = augmentation(image, training=True)
            image = tf.image.random_brightness(image, max_delta=0.15)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_cnn(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = inputs

    for block_index, filters in enumerate((32, 64, 128, 192), start=1):
        for conv_index in (1, 2):
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
                name=f"block{block_index}_conv{conv_index}",
            )(x)
            x = tf.keras.layers.BatchNormalization(
                name=f"block{block_index}_bn{conv_index}"
            )(x)
            x = tf.keras.layers.Activation(
                "relu",
                name=f"block{block_index}_relu{conv_index}",
            )(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            name=f"block{block_index}_pool",
        )(x)
        x = tf.keras.layers.Dropout(
            rate=0.1 + block_index * 0.05,
            name=f"block{block_index}_dropout",
        )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(
        256,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        name="dense",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="dense_batch_normalization")(x)
    x = tf.keras.layers.Activation("relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(0.5, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        name="class_probabilities",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="waste_classifier_cnn")


def calculate_class_weights(
    train_samples: list[tuple[str, int]],
    num_classes: int,
) -> dict[int, float]:
    counts = np.bincount(
        [label for _, label in train_samples],
        minlength=num_classes,
    )
    if np.any(counts == 0):
        raise ValueError("Setiap kelas harus memiliki gambar pada data training.")
    total = int(counts.sum())
    return {
        class_index: total / (num_classes * int(count))
        for class_index, count in enumerate(counts)
    }


def save_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, metrics["accuracy"], label="Train")
    axes[0].plot(epochs, metrics["val_accuracy"], label="Validation")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Akurasi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, metrics["loss"], label="Train")
    axes[1].plot(epochs, metrics["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_samples, validation_samples, test_samples, class_names = collect_splits(
        DATASET_DIR
    )
    train_dataset = make_dataset(train_samples, training=True)
    validation_dataset = make_dataset(validation_samples)
    test_dataset = make_dataset(test_samples)

    print(f"Dataset : {DATASET_DIR}")
    print(f"Jumlah kelas: {len(class_names)}")
    for class_index, class_name in enumerate(class_names):
        train_count = sum(label == class_index for _, label in train_samples)
        validation_count = sum(
            label == class_index for _, label in validation_samples
        )
        test_count = sum(label == class_index for _, label in test_samples)
        print(
            f"  {class_name}: train={train_count}, "
            f"validation={validation_count}, test={test_count}"
        )

    model = build_cnn(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = OUTPUT_DIR / "best_cnn_sampah.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / "training_log.csv"),
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=calculate_class_weights(train_samples, len(class_names)),
        callbacks=callbacks,
    )

    save_training_curves(history, OUTPUT_DIR / "training_curves.png")
    with (OUTPUT_DIR / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)

    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=1)
    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
        "best_model": str(best_model_path),
    }
    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, ensure_ascii=False, indent=2)

    print(f"\nModel terbaik: {best_model_path}")
    print(f"Grafik training: {OUTPUT_DIR / 'training_curves.png'}")
    print(f"Akurasi test: {test_accuracy:.4f}")
    print(f"Loss test: {test_loss:.4f}")


if __name__ == "__main__":
    main()
