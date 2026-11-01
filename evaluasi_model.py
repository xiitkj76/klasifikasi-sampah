"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

EVALUASI MODEL CNN KLASIFIKASI SAMPAH  (Tugas Anggota 4 - Bagian 1)

Mengevaluasi model hasil training (default: model baseline dari CNN_Sampah.py)
pada test set yang SAMA PERSIS dengan yang dipakai saat training, karena
pembagian data diambil langsung dari fungsi collect_splits() di CNN_Sampah.py.

Output (folder hasil_eksperimen/<nama>/):
  metrics.json                 ringkasan angka (dipakai untuk tabel perbandingan)
  classification_report.txt    precision, recall, F1 per kelas
  confusion_matrix.png         confusion matrix jumlah + ternormalisasi
  f1_per_kelas.png             grafik F1-score tiap kelas
  pasangan_tertukar.csv        pasangan kelas yang paling sering tertukar
  akurasi_per_sumber.csv       akurasi untuk gambar asal TrashNet vs RealWaste
  contoh_salah_klasifikasi.png gambar yang salah dengan keyakinan tertinggi
  gradcam.png                  peta perhatian model (Grad-CAM)
  diagnosis_training.txt       analisis kurva training (jika ada training_log.csv)

Cara pakai (jalankan dari folder yang sama dengan CNN_Sampah.py):
  py evaluasi_model.py
  py evaluasi_model.py --model hasil_eksperimen/E2_mobilenetv2/best_model.keras --nama E2_mobilenetv2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import CNN_Sampah as base

BASE_DIR = Path(__file__).resolve().parent
HASIL_DIR = BASE_DIR / "hasil_eksperimen"
DEFAULT_MODEL = BASE_DIR / "cnn_training_output" / "best_cnn_sampah.keras"
DEFAULT_LOG = BASE_DIR / "cnn_training_output" / "training_log.csv"


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def sumber_gambar(path: str) -> str:
    """RealWaste memakai nama 'Kelas_123', TrashNet memakai 'kelas123'."""
    return "RealWaste" if re.search(r"_\d+$", Path(path).stem) else "TrashNet"


def prediksi(model: tf.keras.Model, samples: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Mengembalikan (y_true, probabilitas) dengan urutan sama seperti samples."""
    dataset = base.make_dataset(samples, training=False)  # tanpa shuffle
    probs = model.predict(dataset, verbose=1)
    y_true = np.array([label for _, label in samples])
    return y_true, probs


def cross_entropy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Loss CE murni (tanpa L2/label smoothing) agar adil antar eksperimen."""
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0)
    return float(-np.mean(np.log(p)))


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y in row for y, row in zip(y_true, top_k)]))


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path, judul: str) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    for ax, data, title, fmt in (
        (axes[0], cm, "Jumlah gambar", "d"),
        (axes[1], cm_norm, "Ternormalisasi per kelas asli (recall)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        batas = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > batas else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Confusion Matrix - {judul}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_f1(class_names: list[str], precision, recall, f1, path: Path, judul: str) -> None:
    x = np.arange(len(class_names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, precision, w, label="Precision")
    ax.bar(x, recall, w, label="Recall")
    ax.bar(x + w, f1, w, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Skor")
    ax.set_title(f"Metrik per kelas - {judul}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_salah_klasifikasi(samples, y_true, probs, class_names, path: Path, jumlah: int = 20) -> None:
    y_pred = probs.argmax(axis=1)
    salah = np.where(y_pred != y_true)[0]
    if len(salah) == 0:
        return
    # urutkan dari yang paling "yakin tapi salah"
    salah = salah[np.argsort(-probs[salah, y_pred[salah]])][:jumlah]
    kolom = 5
    baris = int(np.ceil(len(salah) / kolom))
    fig, axes = plt.subplots(baris, kolom, figsize=(3.3 * kolom, 3.9 * baris), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), salah):
        ax.imshow(Image.open(samples[idx][0]).convert("RGB"))
        ax.set_title(
            f"Asli: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]} "
            f"({probs[idx, y_pred[idx]]:.2f})\n[{sumber_gambar(samples[idx][0])}]",
            fontsize=8.5, color="crimson",
        )
    fig.suptitle("Kesalahan dengan keyakinan tertinggi", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cari_layer_konvolusi_terakhir(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.MaxPooling2D)):
            continue
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if len(shape) == 4:
            return layer
    return None


def plot_gradcam(model, samples, y_true, probs, class_names, path: Path, jumlah: int = 10) -> None:
    """Grad-CAM: bagian gambar mana yang paling memengaruhi keputusan model.
    Jika area panas ada di latar belakang (bukan objek), model memakai jalan pintas."""
    layer = cari_layer_konvolusi_terakhir(model)
    if layer is None:
        print("Grad-CAM dilewati: layer konvolusi tidak ditemukan.")
        return
    lapisan = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]

    def forward(batch):
        """Forward pass berurutan (model kita berbentuk rantai lurus), sambil
        menyimpan output layer konvolusi terakhir. Cara ini juga berlaku untuk
        backbone bersarang seperti MobileNetV2 di Keras 3."""
        z, conv_out = batch, None
        for l in lapisan:
            z = l(z, training=False)
            if l is layer:
                conv_out = z
        return conv_out, z

    y_pred = probs.argmax(axis=1)
    rng = np.random.default_rng(0)
    benar = np.where(y_pred == y_true)[0]
    salah = np.where(y_pred != y_true)[0]
    pilih = list(rng.choice(benar, min(jumlah // 2, len(benar)), replace=False)) + list(
        rng.choice(salah, min(jumlah - jumlah // 2, len(salah)), replace=False)
    )

    fig, axes = plt.subplots(2, len(pilih), figsize=(2.6 * len(pilih), 6), squeeze=False)
    for col, idx in enumerate(pilih):
        image, _ = base._load_image(tf.constant(samples[idx][0]), tf.constant(0))
        batch = image[tf.newaxis, ...]
        with tf.GradientTape() as tape:
            conv_out, pred = forward(batch)
            skor = pred[:, int(y_pred[idx])]
        grads = tape.gradient(skor, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., tf.newaxis], base.IMAGE_SIZE).numpy()[..., 0]

        warna = "green" if y_pred[idx] == y_true[idx] else "crimson"
        axes[0, col].imshow(image.numpy())
        axes[0, col].set_title(
            f"Asli: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            fontsize=8, color=warna,
        )
        axes[1, col].imshow(image.numpy())
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle("Grad-CAM (baris bawah: area yang paling diperhatikan model)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analisis kurva training
# ---------------------------------------------------------------------------
def diagnosis_training(log_path: Path) -> tuple[dict, str]:
    with log_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    acc = np.array([float(r["accuracy"]) for r in rows])
    val_acc = np.array([float(r["val_accuracy"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    val_loss = np.array([float(r["val_loss"]) for r in rows])

    best = int(np.argmin(val_loss))
    lonjakan = int(np.sum(np.diff(val_acc) < -0.15))  # turun >15 poin dalam 1 epoch
    ringkas = {
        "epochs_run": len(rows),
        "best_epoch": best + 1,
        "train_acc_at_best": float(acc[best]),
        "val_acc_at_best": float(val_acc[best]),
        "gap_acc_at_best": float(acc[best] - val_acc[best]),
        "val_acc_std_last10": float(np.std(val_acc[-10:])),
        "val_loss_max": float(val_loss.max()),
        "jumlah_lonjakan_val": lonjakan,
    }

    teks = [
        "DIAGNOSIS KURVA TRAINING",
        "=" * 60,
        f"Jumlah epoch berjalan        : {len(rows)}",
        f"Epoch terbaik (val_loss min) : {best + 1}",
        f"Akurasi train / val (terbaik): {acc[best]:.4f} / {val_acc[best]:.4f}",
        f"Selisih (gap) train - val    : {acc[best] - val_acc[best]:.4f}",
        f"Std val_acc 10 epoch terakhir: {np.std(val_acc[-10:]):.4f}",
        f"val_loss tertinggi           : {val_loss.max():.4f} (epoch {int(np.argmax(val_loss)) + 1})",
        f"Penurunan val_acc > 15 poin  : {lonjakan} kali",
        "",
        "Interpretasi:",
    ]
    if lonjakan >= 3 or np.std(val_acc[-10:]) > 0.04:
        teks.append(
            "- Validasi TIDAK STABIL: val_acc naik-turun tajam antar epoch, sedangkan "
            "train_acc naik mulus. Pola ini khas statistik BatchNormalization "
            "(moving mean/variance) yang belum stabil karena learning rate terlalu "
            "besar untuk model yang dilatih dari nol."
        )
    if acc[best] - val_acc[best] > 0.08:
        teks.append(
            "- Ada gap train-val yang cukup besar -> indikasi overfitting ringan; "
            "augmentasi/regularisasi perlu diperkuat."
        )
    if acc.max() < 0.75:
        teks.append(
            f"- Akurasi TRAIN sendiri hanya {acc.max():.2f}: model juga UNDERFITTING. "
            "Kapasitas/fitur yang dipelajari dari nol pada ~4.000 gambar belum cukup; "
            "transfer learning dari ImageNet adalah perbaikan paling berdampak."
        )
    return ringkas, "\n".join(teks)


# ---------------------------------------------------------------------------
# Fungsi utama evaluasi (juga dipanggil oleh eksperimen_perbaikan.py)
# ---------------------------------------------------------------------------
def evaluasi_lengkap(
    model: tf.keras.Model,
    nama: str,
    deskripsi: str = "",
    log_path: Path | None = None,
    info_tambahan: dict | None = None,
    gradcam: bool = True,
) -> dict:
    out = HASIL_DIR / nama
    out.mkdir(parents=True, exist_ok=True)

    _, val_samples, test_samples, class_names = base.collect_splits(base.DATASET_DIR)

    print(f"\n[{nama}] Prediksi validation set...")
    y_val, p_val = prediksi(model, val_samples)
    print(f"[{nama}] Prediksi test set...")
    y_true, probs = prediksi(model, test_samples)
    y_pred = probs.argmax(axis=1)

    labels = list(range(len(class_names)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- simpan teks & grafik
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(cm, class_names, out / "confusion_matrix.png", nama)
    plot_f1(class_names, prec, rec, f1, out / "f1_per_kelas.png", nama)
    plot_salah_klasifikasi(test_samples, y_true, probs, class_names, out / "contoh_salah_klasifikasi.png")
    if gradcam:
        try:
            plot_gradcam(model, test_samples, y_true, probs, class_names, out / "gradcam.png")
        except Exception as error:  # Grad-CAM bersifat tambahan, jangan gagalkan evaluasi
            print(f"Grad-CAM dilewati: {error}")

    # --- pasangan kelas yang paling sering tertukar
    pasangan = [
        (class_names[i], class_names[j], int(cm[i, j]), cm[i, j] / max(cm[i].sum(), 1))
        for i in labels for j in labels if i != j and cm[i, j] > 0
    ]
    pasangan.sort(key=lambda r: -r[2])
    with (out / "pasangan_tertukar.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label_asli", "diprediksi_sebagai", "jumlah", "persen_dari_kelas_asli"])
        for a, b, n, p in pasangan:
            w.writerow([a, b, n, f"{p:.1%}"])

    # --- akurasi berdasarkan sumber dataset (deteksi jalan pintas/shortcut)
    sumber = np.array([sumber_gambar(p) for p, _ in test_samples])
    with (out / "akurasi_per_sumber.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kelas", "sumber", "jumlah_test", "akurasi"])
        for c, name in enumerate(class_names):
            for s in ("TrashNet", "RealWaste"):
                mask = (y_true == c) & (sumber == s)
                if mask.sum():
                    w.writerow([name, s, int(mask.sum()), f"{np.mean(y_pred[mask] == c):.4f}"])
        for s in ("TrashNet", "RealWaste"):
            mask = sumber == s
            w.writerow(["SEMUA", s, int(mask.sum()), f"{np.mean(y_pred[mask] == y_true[mask]):.4f}"])

    metrics = {
        "nama": nama,
        "deskripsi": deskripsi,
        "test_accuracy": float(np.mean(y_pred == y_true)),
        "test_top3_accuracy": top_k_accuracy(y_true, probs, 3),
        "test_cross_entropy": cross_entropy(y_true, probs),
        "test_macro_precision": float(np.mean(prec)),
        "test_macro_recall": float(np.mean(rec)),
        "test_macro_f1": float(np.mean(f1)),
        "test_weighted_f1": float(np.average(f1, weights=support)),
        "val_accuracy": float(np.mean(p_val.argmax(axis=1) == y_val)),
        "f1_per_kelas": {n: float(v) for n, v in zip(class_names, f1)},
        "total_params": int(model.count_params()),
    }
    if log_path and Path(log_path).is_file():
        ringkas, teks = diagnosis_training(Path(log_path))
        metrics.update(ringkas)
        (out / "diagnosis_training.txt").write_text(teks, encoding="utf-8")
        print("\n" + teks)
    if info_tambahan:
        metrics.update(info_tambahan)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- cetak ringkasan
    print("\n" + "=" * 72)
    print(f"HASIL EVALUASI: {nama}")
    print("=" * 72)
    print(report)
    print(f"Akurasi test        : {metrics['test_accuracy']:.4f}")
    print(f"Top-3 accuracy      : {metrics['test_top3_accuracy']:.4f}")
    print(f"Macro F1            : {metrics['test_macro_f1']:.4f}")
    print(f"Weighted F1         : {metrics['test_weighted_f1']:.4f}")
    print(f"Akurasi validation  : {metrics['val_accuracy']:.4f}")
    print("\n5 pasangan kelas paling sering tertukar:")
    for a, b, n, p in pasangan[:5]:
        print(f"  {a:>20s} -> {b:<20s} {n:3d} gambar ({p:.0%})")
    print(f"\nSemua output tersimpan di: {out}")
    return metrics


def muat_model(path: Path) -> tf.keras.Model:
    """Membuka model .keras. Jika gagal karena versi Keras berbeda (misalnya model
    disimpan dengan Keras yang lebih baru dari Keras di Colab), arsitektur dibangun
    ulang dari CNN_Sampah.build_cnn() lalu hanya bobotnya yang dimuat."""
    try:
        return tf.keras.models.load_model(str(path), compile=False)
    except (TypeError, ValueError) as error:
        print("\n[INFO] Model tidak bisa dibuka langsung (kemungkinan versi Keras berbeda):")
        print(f"       {str(error).splitlines()[0][:150]}")
        print("[INFO] Membangun ulang arsitektur dari CNN_Sampah.build_cnn() dan memuat bobotnya...")
        _, _, _, class_names = base.collect_splits(base.DATASET_DIR)
        model = base.build_cnn(num_classes=len(class_names))
        model.load_weights(str(path))
        print("[INFO] Bobot berhasil dimuat.\n")
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi model CNN klasifikasi sampah")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--nama", default="E0_baseline")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="training_log.csv (opsional)")
    parser.add_argument("--deskripsi", default="CNN 4 blok dari nol (model Anggota 3)")
    parser.add_argument("--tanpa-gradcam", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(base.SEED)
    model = muat_model(args.model)
    metrics = evaluasi_lengkap(
        model, args.nama, args.deskripsi, log_path=args.log, gradcam=not args.tanpa_gradcam
    )

    # Pengecekan: akurasi baseline harus sama dengan hasil training Anggota 3
    acuan = Path(args.model).parent / "test_metrics.json"
    if acuan.is_file():
        akurasi_acuan = json.loads(acuan.read_text(encoding="utf-8"))["test_accuracy"]
        selisih = abs(metrics["test_accuracy"] - akurasi_acuan)
        print(f"\nAkurasi test di test_metrics.json : {akurasi_acuan:.4f}")
        print(f"Akurasi test hasil evaluasi ini   : {metrics['test_accuracy']:.4f}")
        print("Status: SAMA, model & test set terbukti identik" if selisih < 0.005 else
              "PERINGATAN: akurasi berbeda. Periksa apakah dataset/model sama dengan milik Anggota 3.")


if __name__ == "__main__":
    main()
