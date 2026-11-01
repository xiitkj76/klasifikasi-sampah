"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

EKSPERIMEN PERBAIKAN MODEL  (Tugas Anggota 4 - Bagian 2)

Masalah yang terlihat pada model baseline (CNN_Sampah.py):
  1. Validasi sangat tidak stabil (val_acc 0.41 -> 0.10 -> 0.17 -> 0.14, val_loss
     sampai 6.87) padahal akurasi train naik mulus. Ini khas statistik
     BatchNormalization yang tertinggal karena learning rate 1e-3 terlalu besar.
  2. Akurasi train sendiri hanya ~0.68 -> model underfitting; fitur yang
     dipelajari dari nol dengan ~4.000 gambar belum cukup kuat.
  3. Akurasi test 0.554.

Daftar eksperimen (semua memakai split train/val/test yang sama dengan baseline):
  E1_cnn_stabil        Arsitektur CNN yang SAMA, tetapi training distabilkan:
                       AdamW + warmup + cosine decay (LR 5e-4), BN momentum 0.9,
                       label smoothing 0.1, augmentasi lebih kuat.
  E2_mobilenetv2       Transfer learning MobileNetV2 (ImageNet): latih head dulu,
                       lalu fine-tuning 40 layer teratas dengan LR kecil.
  E3_efficientnetb0    Sama seperti E2 dengan backbone EfficientNetB0 (opsional).
  A1..A4               Studi ablasi (sesuai judul: augmentasi & regularisasi)
                       dengan backbone MobileNetV2 beku agar cepat:
                         A1 tanpa augmentasi, tanpa regularisasi
                         A2 dengan augmentasi saja
                         A3 dengan regularisasi saja (dropout+weight decay+label smoothing)
                         A4 augmentasi + regularisasi
                         A5 sama dengan A3 tetapi TANPA class weight
                            (bukti dampak penanganan data tidak seimbang)

Cara pakai (dari folder yang sama dengan CNN_Sampah.py):
  py eksperimen_perbaikan.py --daftar
  py eksperimen_perbaikan.py --exp E2 --uji-cepat     (cek skrip jalan, ~beberapa menit)
  py eksperimen_perbaikan.py --exp E2
  py eksperimen_perbaikan.py --exp A1 A2 A3 A4 A5
  py eksperimen_perbaikan.py --exp E1
  py eksperimen_perbaikan.py --ringkasan              (tabel + grafik perbandingan)

Catatan: E2/E3/A* mengunduh bobot ImageNet sekali saat pertama dijalankan
(butuh internet, ~9 MB MobileNetV2 / ~16 MB EfficientNetB0).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import CNN_Sampah as base
from evaluasi_model import HASIL_DIR, evaluasi_lengkap

AUTOTUNE = tf.data.AUTOTUNE

REG = dict(dropout=0.3, weight_decay=1e-4, label_smoothing=0.1)
TANPA_REG = dict(dropout=0.0, weight_decay=0.0, label_smoothing=0.0)
ABLASI = dict(jenis="transfer", backbone="mobilenetv2", lr=1e-3, epochs=15, patience=5,
              fine_tune_epochs=0)

EKSPERIMEN: dict[str, dict] = {
    "E1_cnn_stabil": dict(
        jenis="scratch",
        deskripsi="CNN baseline + AdamW, warmup-cosine LR 5e-4, BN momentum 0.9, "
                  "label smoothing 0.1, augmentasi kuat",
        augmentasi=True, lr=5e-4, warmup_epochs=3, epochs=60, patience=12,
        bn_momentum=0.9, weight_decay=1e-4, label_smoothing=0.1,
    ),
    "E2_mobilenetv2": dict(
        jenis="transfer", backbone="mobilenetv2",
        deskripsi="Transfer learning MobileNetV2 + augmentasi + regularisasi, fine-tuning 40 layer",
        augmentasi=True, lr=1e-3, epochs=15, patience=5,
        fine_tune_epochs=15, fine_tune_lr=1e-5, fine_tune_layers=40, **REG,
    ),
    "E3_efficientnetb0": dict(
        jenis="transfer", backbone="efficientnetb0",
        deskripsi="Transfer learning EfficientNetB0 + augmentasi + regularisasi, fine-tuning 40 layer",
        augmentasi=True, lr=1e-3, epochs=15, patience=5,
        fine_tune_epochs=15, fine_tune_lr=1e-5, fine_tune_layers=40, **REG,
    ),
    "A1_tanpa_aug_tanpa_reg": dict(
        deskripsi="Ablasi: MobileNetV2 beku, tanpa augmentasi, tanpa regularisasi",
        augmentasi=False, **ABLASI, **TANPA_REG,
    ),
    "A2_aug_saja": dict(
        deskripsi="Ablasi: MobileNetV2 beku, augmentasi saja",
        augmentasi=True, **ABLASI, **TANPA_REG,
    ),
    "A3_reg_saja": dict(
        deskripsi="Ablasi: MobileNetV2 beku, regularisasi saja",
        augmentasi=False, **ABLASI, **REG,
    ),
    "A4_aug_dan_reg": dict(
        deskripsi="Ablasi: MobileNetV2 beku, augmentasi + regularisasi",
        augmentasi=True, **ABLASI, **REG,
    ),
    "A5_reg_tanpa_class_weight": dict(
        deskripsi="Ablasi: MobileNetV2 beku, regularisasi, TANPA class weight",
        augmentasi=False, class_weight=False, **ABLASI, **REG,
    ),
}


# ---------------------------------------------------------------------------
# Input pipeline (preprocessing identik dengan baseline: piksel 0-1, 224x224)
# ---------------------------------------------------------------------------
def _baca_uint8(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, base.IMAGE_SIZE)
    return tf.saturate_cast(tf.round(image), tf.uint8), label


def buat_augmentasi():
    """Augmentasi per gambar dalam satu batch (geometri + warna)."""
    geometri = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=base.SEED),
            tf.keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=base.SEED + 1),
            tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=base.SEED + 2),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=base.SEED + 3),
        ],
        name="augmentasi_geometri",
    )

    def augment(images):
        images = geometri(images, training=True)
        n = tf.shape(images)[0]
        # kecerahan, kontras, dan saturasi acak per gambar
        images = images + tf.random.uniform([n, 1, 1, 1], -0.15, 0.15)
        mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
        images = (images - mean) * tf.random.uniform([n, 1, 1, 1], 0.8, 1.2) + mean
        gray = tf.image.rgb_to_grayscale(images)
        images = gray + (images - gray) * tf.random.uniform([n, 1, 1, 1], 0.7, 1.3)
        return tf.clip_by_value(images, 0.0, 1.0)

    return augment


def buat_dataset(samples, num_classes, training=False, augmentasi=False, class_weights=None):
    paths, labels = zip(*samples)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(_baca_uint8, num_parallel_calls=AUTOTUNE).cache()  # ~600 MB RAM untuk train
    if training:
        ds = ds.shuffle(len(samples), seed=base.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(base.BATCH_SIZE)

    augment = buat_augmentasi() if (training and augmentasi) else None
    bobot = None
    if training and class_weights:
        bobot = tf.constant([class_weights[i] for i in range(num_classes)], tf.float32)

    def akhir(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        if augment is not None:
            images = augment(images)
        onehot = tf.one_hot(labels, num_classes)
        if bobot is not None:
            return images, onehot, tf.gather(bobot, labels)  # class weight sbg sample weight
        return images, onehot

    return ds.map(akhir, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_cnn_stabil(num_classes: int, bn_momentum: float) -> tf.keras.Model:
    """Arsitektur sama persis dengan baseline; perbedaan: BN momentum dan tanpa L2
    (regularisasi bobot dipindah ke AdamW weight decay)."""
    inputs = tf.keras.Input(shape=(*base.IMAGE_SIZE, 3), name="image")
    x = inputs
    for b, filters in enumerate((32, 64, 128, 192), start=1):
        for c in (1, 2):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                       name=f"block{b}_conv{c}")(x)
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name=f"block{b}_bn{c}")(x)
            x = tf.keras.layers.Activation("relu", name=f"block{b}_relu{c}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"block{b}_pool")(x)
        x = tf.keras.layers.Dropout(0.1 + b * 0.05, name=f"block{b}_dropout")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, use_bias=False, name="dense")(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name="dense_batch_normalization")(x)
    x = tf.keras.layers.Activation("relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(0.5, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_sampah_stabil")


def build_transfer(num_classes: int, backbone: str, dropout: float):
    inputs = tf.keras.Input(shape=(*base.IMAGE_SIZE, 3), name="image")  # piksel 0-1
    if backbone == "mobilenetv2":
        x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="ke_rentang_min1_1")(inputs)
        backbone_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*base.IMAGE_SIZE, 3))
    elif backbone == "efficientnetb0":
        x = tf.keras.layers.Rescaling(255.0, name="ke_rentang_0_255")(inputs)
        backbone_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*base.IMAGE_SIZE, 3))
    else:
        raise ValueError(f"Backbone tidak dikenal: {backbone}")
    backbone_model.trainable = False
    x = backbone_model(x, training=False)  # BN backbone selalu mode inference
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name=f"tl_{backbone}"), backbone_model


def compile_model(model, learning_rate, weight_decay, label_smoothing):
    if weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_fase(model, train_ds, val_ds, out, epochs, initial_epoch, patience,
             threshold=None, reduce_lr=True):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True,
            initial_value_threshold=threshold, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out / "training_log.csv"), append=initial_epoch > 0),
    ]
    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1))
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch, callbacks=callbacks,
    )
    return len(history.history["loss"]), callbacks[0].best


def plot_kurva(log_path: Path, path: Path, judul: str, garis_fine_tune: int | None = None):
    with log_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) + 1 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], "accuracy", "Akurasi"), (axes[1], "loss", "Loss")):
        ax.plot(ep, [float(r[key]) for r in rows], label="Train")
        ax.plot(ep, [float(r["val_" + key]) for r in rows], label="Validation")
        if garis_fine_tune:
            ax.axvline(garis_fine_tune + 0.5, color="gray", ls="--", label="Mulai fine-tuning")
        ax.set_title(f"{title} - {judul}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def jalankan(nama: str, cfg: dict, uji_cepat: bool = False) -> dict:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(base.SEED)
    nama_out = f"_uji_{nama}" if uji_cepat else nama
    out = HASIL_DIR / nama_out
    out.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 72 + f"\n# EKSPERIMEN {nama}\n# {cfg['deskripsi']}\n" + "#" * 72)

    train_s, val_s, _, class_names = base.collect_splits(base.DATASET_DIR)
    if uji_cepat:  # 24 gambar/kelas, 1 epoch per fase: hanya untuk memastikan skrip berjalan
        train_s = [s for c in range(len(class_names)) for s in [x for x in train_s if x[1] == c][:24]]
        cfg = {**cfg, "epochs": 1, "fine_tune_epochs": min(1, cfg.get("fine_tune_epochs", 0)),
               "warmup_epochs": 0}
    n_cls = len(class_names)
    class_weights = (base.calculate_class_weights(train_s, n_cls)
                     if cfg.get("class_weight", True) else None)
    train_ds = buat_dataset(train_s, n_cls, training=True, augmentasi=cfg["augmentasi"],
                            class_weights=class_weights)
    val_ds = buat_dataset(val_s, n_cls)
    steps = math.ceil(len(train_s) / base.BATCH_SIZE)

    mulai = time.time()
    garis_ft = None
    if cfg["jenis"] == "scratch":
        model = build_cnn_stabil(n_cls, cfg["bn_momentum"])
        warmup = cfg["warmup_epochs"] * steps
        total = cfg["epochs"] * steps
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg["lr"] * 0.1 if warmup else cfg["lr"],
            decay_steps=max(total - warmup, 1), alpha=0.01,
            warmup_target=cfg["lr"] if warmup else None, warmup_steps=warmup,
        )
        compile_model(model, lr, cfg["weight_decay"], cfg["label_smoothing"])
        model.summary()
        epochs_run, _ = fit_fase(model, train_ds, val_ds, out, cfg["epochs"], 0,
                                 cfg["patience"], reduce_lr=False)
    else:
        model, backbone = build_transfer(n_cls, cfg["backbone"], cfg["dropout"])
        compile_model(model, cfg["lr"], cfg["weight_decay"], cfg["label_smoothing"])
        model.summary()
        print("\n--- Fase 1: melatih classifier head (backbone dibekukan) ---")
        epochs_run, best = fit_fase(model, train_ds, val_ds, out, cfg["epochs"], 0, cfg["patience"])

        if cfg.get("fine_tune_epochs", 0) > 0:
            print(f"\n--- Fase 2: fine-tuning {cfg['fine_tune_layers']} layer teratas ---")
            model.load_weights(str(out / "best.weights.h5"))
            backbone.trainable = True
            for layer in backbone.layers[: -cfg["fine_tune_layers"]]:
                layer.trainable = False
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            compile_model(model, cfg["fine_tune_lr"], cfg["weight_decay"], cfg["label_smoothing"])
            garis_ft = epochs_run
            n2, _ = fit_fase(model, train_ds, val_ds, out, cfg["fine_tune_epochs"], epochs_run,
                             cfg["patience"], threshold=best)
            epochs_run += n2

    menit = (time.time() - mulai) / 60
    model.load_weights(str(out / "best.weights.h5"))
    model.save(str(out / "best_model.keras"))
    plot_kurva(out / "training_log.csv", out / "training_curves.png", nama, garis_ft)

    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    return evaluasi_lengkap(
        model, nama_out, cfg["deskripsi"], log_path=out / "training_log.csv",
        info_tambahan={"waktu_training_menit": round(menit, 1), "trainable_params_akhir": trainable,
                       "konfigurasi": {k: v for k, v in cfg.items() if k != "deskripsi"}},
    )


# ---------------------------------------------------------------------------
# Ringkasan perbandingan semua eksperimen
# ---------------------------------------------------------------------------
def ringkasan() -> None:
    hasil = []
    for path in sorted(HASIL_DIR.glob("*/metrics.json")):
        if path.parent.name.startswith("_"):
            continue  # lewati hasil uji cepat
        hasil.append(json.loads(path.read_text(encoding="utf-8")))
    if not hasil:
        print("Belum ada hasil. Jalankan evaluasi_model.py dan/atau eksperimen dulu.")
        return

    kolom = ["nama", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_top3_accuracy",
             "test_cross_entropy", "val_accuracy", "best_epoch", "epochs_run",
             "waktu_training_menit", "total_params", "deskripsi"]
    with (HASIL_DIR / "perbandingan_eksperimen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(kolom)
        for h in hasil:
            w.writerow([h.get(k, "") for k in kolom])

    baseline = next((h for h in hasil if h["nama"].startswith("E0")), None)
    md = ["| Eksperimen | Akurasi test | Macro F1 | Top-3 | Δ akurasi vs baseline | Keterangan |",
          "|---|---|---|---|---|---|"]
    for h in hasil:
        delta = f"{(h['test_accuracy'] - baseline['test_accuracy']) * 100:+.1f} poin" if baseline else "-"
        md.append(f"| {h['nama']} | {h['test_accuracy']:.4f} | {h['test_macro_f1']:.4f} | "
                  f"{h['test_top3_accuracy']:.4f} | {delta} | {h.get('deskripsi', '')} |")
    (HASIL_DIR / "perbandingan_eksperimen.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # grafik akurasi & macro F1
    nama = [h["nama"] for h in hasil]
    x = np.arange(len(nama))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(nama)), 5))
    b1 = ax.bar(x - 0.2, [h["test_accuracy"] for h in hasil], 0.4, label="Akurasi test")
    b2 = ax.bar(x + 0.2, [h["test_macro_f1"] for h in hasil], 0.4, label="Macro F1")
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(nama, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan eksperimen (test set)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "perbandingan_eksperimen.png", dpi=130)
    plt.close(fig)

    # heatmap F1 per kelas
    kelas = list(hasil[0]["f1_per_kelas"].keys())
    data = np.array([[h["f1_per_kelas"][k] for k in kelas] for h in hasil])
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(hasil) + 2.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kelas)))
    ax.set_xticklabels(kelas, rotation=30, ha="right")
    ax.set_yticks(range(len(nama)))
    ax.set_yticklabels(nama)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("F1-score per kelas untuk setiap eksperimen")
    fig.tight_layout()
    fig.savefig(HASIL_DIR / "f1_per_kelas_perbandingan.png", dpi=130)
    plt.close(fig)
    print(f"\nRingkasan disimpan di: {HASIL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperimen perbaikan model klasifikasi sampah")
    parser.add_argument("--exp", nargs="+", help="Nama/prefix eksperimen, mis. E2 atau A1 A2 A3 A4")
    parser.add_argument("--uji-cepat", action="store_true", help="Subset kecil, 1 epoch (cek skrip)")
    parser.add_argument("--ringkasan", action="store_true", help="Buat tabel & grafik perbandingan")
    parser.add_argument("--daftar", action="store_true", help="Tampilkan daftar eksperimen")
    args = parser.parse_args()

    if args.daftar or not (args.exp or args.ringkasan):
        for k, v in EKSPERIMEN.items():
            print(f"{k:26s} {v['deskripsi']}")
        return
    for prefix in args.exp or []:
        cocok = [k for k in EKSPERIMEN if k == prefix or k.startswith(prefix + "_")]
        if not cocok:
            raise SystemExit(f"Eksperimen '{prefix}' tidak ditemukan. Pakai --daftar.")
        jalankan(cocok[0], EKSPERIMEN[cocok[0]], uji_cepat=args.uji_cepat)
    if args.ringkasan or (args.exp and not args.uji_cepat):
        ringkasan()


if __name__ == "__main__":
    main()
