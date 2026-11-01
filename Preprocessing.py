from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import hashlib
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import TypeAlias, Union
import zipfile
from PIL import Image
import tensorflow as tf


DATASET_DIR = Path(__file__).resolve().parent / "datasett"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

Sample: TypeAlias = tuple[str, int]

def inspect_dataset(data_dir: Union[str, Path]):
    "untuk memeriksa gambar rusak dan file gambar duplikat."

    dataset_dir = Path(data_dir)
    corrupt_images = []
    duplicate_images = []
    excluded_paths = set()
    seen_hashes = {}

    image_files = sorted(
        (
            path
            for path in dataset_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )

    for path in image_files:
        try:
            image_bytes = tf.io.read_file(str(path))
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )

            # Memastikan gambar memiliki ukuran yang valid.
            shape = tf.shape(image).numpy()
            if shape[0] <= 0 or shape[1] <= 0:
                raise ValueError("Ukuran gambar tidak valid")

            # Hash berdasarkan isi file untuk mencari duplikat identik.
            image_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            if image_hash in seen_hashes:
                original_path = seen_hashes[image_hash]
                duplicate_images.append(
                    (str(path), original_path)
                )
                excluded_paths.add(str(path))
            else:
                seen_hashes[image_hash] = str(path)

        except (
            tf.errors.OpError,
            ValueError,
            OSError,
        ) as error:
            corrupt_images.append((str(path), str(error)))
            excluded_paths.add(str(path))

    return corrupt_images, duplicate_images, excluded_paths


def show_preprocessing_samples(
    samples: list[Sample],
    number_of_samples: int = 3,
):
    "Menampilkan gambar asli dan hasil preprocessing."

    if not samples:
        print("Tidak ada gambar untuk ditampilkan.")
        return

    number_of_samples = min(
        number_of_samples, len(samples)
    )

    fig, axes = plt.subplots(
        number_of_samples,
        2,
        figsize=(10, 4 * number_of_samples),
        squeeze=False,
    )

    for index in range(number_of_samples):
        image_path, label = samples[index]

        # Gambar asli.
        original = tf.io.decode_image(
            tf.io.read_file(image_path),
            channels=3,
            expand_animations=False,
        )

        # Gambar setelah resize dan normalisasi.
        processed, _ = preprocess_image(
            tf.constant(image_path),
            tf.constant(label),
        )

        axes[index, 0].imshow(original.numpy())
        axes[index, 0].set_title(
            f"Asli: {CATEGORIES[label]}"
        )
        axes[index, 0].axis("off")

        axes[index, 1].imshow(processed.numpy())
        axes[index, 1].set_title(
            f"Preprocessing: {CATEGORIES[label]} "
            f"(label {label})"
        )
        axes[index, 1].axis("off")

        print(
            f"Contoh {index + 1}: "
            f"{Path(image_path).name} | "
            f"Kategori: {CATEGORIES[label]} | "
            f"Label: {label} | "
            f"Ukuran hasil: {tuple(processed.shape)} | "
            f"Rentang piksel: "
            f"{float(tf.reduce_min(processed).numpy()):.4f}"
            f" - "
            f"{float(tf.reduce_max(processed).numpy()):.4f}"
        )

    plt.tight_layout()
    plt.show()



def _collect_splits(
    data_dir: Union[str, Path], seed: int
) -> tuple[list[Sample], list[Sample], list[Sample], list[Sample], list[str]]:
    dataset_dir = Path(data_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    class_dirs = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas di: {dataset_dir}")

    all_samples: list[Sample] = []
    train_samples: list[Sample] = []
    validation_samples: list[Sample] = []
    test_samples: list[Sample] = []
    class_names: list[str] = []

    for class_index, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        image_paths = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() 
                and path.suffix.lower() in VALID_EXTENSIONS
                and str(path) not in EXCLUDED_PATHS
            ),
            key=lambda path: (path.name.casefold(), str(path).casefold()),
        )
        if not image_paths:
            raise ValueError(f"Tidak ada gambar di folder kelas: {class_dir}")

        class_samples = [(str(path), class_index) for path in image_paths]
        all_samples.extend(class_samples)
        random.Random(seed + class_index).shuffle(class_samples)

        train_count = int(len(class_samples) * TRAIN_RATIO)
        validation_count = int(len(class_samples) * VALIDATION_RATIO)
        test_count = len(class_samples) - train_count - validation_count
        if min(train_count, validation_count, test_count) == 0:
            raise ValueError(
                f"Kelas '{class_dir.name}' harus memiliki cukup gambar untuk "
                "mengisi split train, validation, dan test."
            )

        train_samples.extend(class_samples[:train_count])
        validation_samples.extend(
            class_samples[train_count : train_count + validation_count]
        )
        test_samples.extend(class_samples[train_count + validation_count :])

    return all_samples, train_samples, validation_samples, test_samples, class_names


def preprocess_image(
    image_path: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.int32)


def create_pipeline(
    paths: list[str],
    labels: list[int],
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tf.data.Dataset:
    if not paths:
        raise ValueError("Pipeline tidak dapat dibuat dari dataset kosong.")
    if len(paths) != len(labels):
        raise ValueError("Jumlah paths dan labels harus sama.")
    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari nol.")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=seed),
                tf.keras.layers.RandomRotation(0.1, seed=seed + 1),
                tf.keras.layers.RandomZoom(0.1, seed=seed + 2),
            ]
        )
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(
    data_dir: Union[str, Path] = DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    "Return train, validation, and test datasets and the label class names."
    _, train_samples, validation_samples, test_samples, class_names = (
        _collect_splits(data_dir, seed)
    )

    def make_pipeline(samples: list[Sample], training: bool = False) -> tf.data.Dataset:
        paths, labels = zip(*samples)
        return create_pipeline(
            list(paths),
            list(labels),
            training=training,
            batch_size=batch_size,
            seed=seed,
        )

    return (
        make_pipeline(train_samples, training=True),
        make_pipeline(validation_samples),
        make_pipeline(test_samples),
        class_names,
    )


def save_preprocessed_images(samples, class_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    for image_path, label in samples:
        image = preprocess_image(image_path, label)[0].numpy()

        # Ubah nilai piksel 0–1 menjadi 0–255 untuk format PNG
        image = (image * 255).round().astype("uint8")

        class_name = class_names[label]
        output_path = output_dir / class_name / (
            Path(image_path).stem + ".png"
        )

        Image.fromarray(image).save(output_path)

    print(f"Gambar hasil preprocessing disimpan di: {output_dir}")


def zip_preprocessed_dataset(source_dir, zip_path):
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_file:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(
                    file_path,
                    file_path.relative_to(source_dir.parent),
                )

    print(f"File ZIP berhasil dibuat di: {zip_path}")


# Pemeriksaan dilakukan sebelum pembagian dataset.
CORRUPT_IMAGES, DUPLICATE_IMAGES, EXCLUDED_PATHS = (
    inspect_dataset(DATASET_DIR)
)

# Mengumpulkan dan membagi dataset setelah pemeriksaan.
(
    _all_samples,
    _train_samples,
    _validation_samples,
    _test_samples,
    CATEGORIES,
) = _collect_splits(DATASET_DIR, SEED)

image_paths = [path for path, _ in _all_samples]
image_labels = [label for _, label in _all_samples]
X_train = [path for path, _ in _train_samples]
y_train = [label for _, label in _train_samples]
X_val = [path for path, _ in _validation_samples]
y_val = [label for _, label in _validation_samples]
X_test = [path for path, _ in _test_samples]
y_test = [label for _, label in _test_samples]

train_dataset = create_pipeline(X_train, y_train, training=True)
val_dataset = create_pipeline(X_val, y_val)
test_dataset = create_pipeline(X_test, y_test)

def zip_preprocessed_dataset(
    source_dir: Union[str, Path],
    zip_path: Union[str, Path],
):
    "Mengarsipkan seluruh folder hasil preprocessing menjadi ZIP."

    source_dir = Path(source_dir)
    zip_path = Path(zip_path)

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Folder hasil preprocessing tidak ditemukan: {source_dir}"
        )

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0

    with zipfile.ZipFile(
        zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive_name = (
                    Path(source_dir.name) / path.relative_to(source_dir)
                )
                archive.write(path, arcname=str(archive_name))
                file_count += 1

    print("\nHASIL PEMBUATAN FILE ZIP")
    print("-" * 65)
    print(f"Jumlah file gambar : {file_count}")
    print(f"Lokasi file ZIP    : {zip_path.resolve()}")
    print(f"Ukuran ZIP         : {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("Status             : BERHASIL")

    return zip_path


if __name__ == "__main__":
    print("=" * 72)
    print("          HASIL PREPROCESSING DAN INPUT PIPELINE")
    print("=" * 72)

    # A. Ringkasan dataset
    print("\nA. RINGKASAN DATASET")
    print("-" * 72)
    print(f"{'Informasi':<30}{'Jumlah':>15}")
    print("-" * 72)
    print(f"{'Total gambar':<30}{len(image_paths):>15}")
    print(f"{'Jumlah kategori':<30}{len(CATEGORIES):>15}")

    # B. Jumlah gambar setiap kategori
    print("\nB. JUMLAH GAMBAR PER KATEGORI")
    print("-" * 72)
    print(f"{'No.':<6}{'Kategori':<40}{'Jumlah gambar':>20}")
    print("-" * 72)

    for index, category in enumerate(CATEGORIES, start=1):
        jumlah = image_labels.count(index - 1)
        print(f"{index:<6}{category:<40}{jumlah:>20}")

    print("-" * 72)
    print(f"{'Total':<46}{len(image_paths):>20}")

    # C. Pembagian dataset
    print("\nC. PEMBAGIAN DATASET")
    print("-" * 72)
    print(f"{'Dataset':<20}{'Jumlah gambar':>20}{'Persentase':>20}")
    print("-" * 72)

    total = len(image_paths)
    splits = [
        ("Training", len(X_train)),
        ("Validation", len(X_val)),
        ("Testing", len(X_test)),
    ]

    for nama, jumlah in splits:
        persentase = jumlah / total * 100
        print(f"{nama:<20}{jumlah:>20}{persentase:>19.1f}%")

    print("-" * 72)
    print(f"{'Total':<20}{total:>20}{100:>19.1f}%")

    # D. Pemeriksaan input pipeline
    print("\nD. PEMERIKSAAN INPUT PIPELINE")
    print("-" * 72)

    for images, labels in train_dataset.take(1):
        print(f"{'Parameter':<35}{'Hasil':>30}")
        print("-" * 72)
        print(f"{'Ukuran batch gambar':<35}{str(images.shape):>30}")
        print(f"{'Ukuran batch label':<35}{str(labels.shape):>30}")
        print(
            f"{'Nilai piksel minimum':<35}"
            f"{tf.reduce_min(images).numpy():>30.4f}"
        )
        print(
            f"{'Nilai piksel maksimum':<35}"
            f"{tf.reduce_max(images).numpy():>30.4f}"
        )
    
    # E. Pemeriksaan gambar rusak
    print("\nE. PEMERIKSAAN GAMBAR RUSAK")
    print("-" * 72)
    print(f"{'Status':<35}{'Jumlah':>15}")
    print("-" * 72)
    print(f"{'Gambar rusak/tidak terbaca':<35}{len(CORRUPT_IMAGES):>15}")
    print(f"{'Gambar valid sebelum deduplikasi':<35}"
          f"{len(image_paths) + len(CORRUPT_IMAGES) + len(DUPLICATE_IMAGES):>15}")

    if CORRUPT_IMAGES:
        print("\nDaftar gambar rusak:")
        for path, error in CORRUPT_IMAGES:
            print(f"- {path}")
            print(f"  Penyebab: {error}")
    else:
        print("Semua file gambar berhasil didekode.")

    # F. Pemeriksaan gambar duplikat
    print("\nF. PEMERIKSAAN GAMBAR DUPLIKAT")
    print("-" * 72)
    print(f"{'Jumlah file duplikat':<35}{len(DUPLICATE_IMAGES):>15}")

    if DUPLICATE_IMAGES:
        for duplicate, original in DUPLICATE_IMAGES:
            print(f"\nFile duplikat : {duplicate}")
            print(f"File pertama  : {original}")
    else:
        print("Tidak ditemukan file identik berdasarkan SHA-256.")

    print("\nCatatan: pemeriksaan duplikat ini hanya mendeteksi")
    print("file yang isinya identik, bukan gambar yang sekadar mirip.")

    # G. Contoh hasil preprocessing
    print("\nG. CONTOH HASIL PREPROCESSING")
    print("-" * 72)
    show_preprocessing_samples(_all_samples, number_of_samples=3)
    
    #h Simpan Gambar hasil preprocessing
    OUTPUT_DIR = DATASET_DIR.parent / "datasett_preprocessed"

    save_preprocessed_images(
        _all_samples,
        CATEGORIES,
        OUTPUT_DIR,
    )

    # i. Buat ZIP dari hasil preprocessing
    ZIP_PATH = DATASET_DIR.parent / "datasett_preprocessed.zip"

    zip_preprocessed_dataset(
        OUTPUT_DIR,
        ZIP_PATH,
    )

    print("-" * 72)
    print("Status preprocessing dan input pipeline: BERHASIL")
    print("=" * 72)
