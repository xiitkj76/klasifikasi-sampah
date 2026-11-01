"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

MODUL INFERENSI KLASIFIKASI SAMPAH  (dipakai bersama oleh fitur_unik.py,
test_sampah.py, dan aplikasi deployment, sehingga hasilnya selalu konsisten)

Fitur yang disediakan kelas PrediktorSampah:
  - preprocessing identik dengan training (resize 224x224 bilinear, piksel 0-1)
  - prediksi top-3 beserta probabilitas
  - confidence threshold  -> status "tidak_yakin" jika keyakinan rendah
  - deteksi input di luar domain (bukan foto sampah) dengan kemiripan kosinus
    fitur gambar terhadap rata-rata fitur (centroid) tiap kelas pada data training
  - Grad-CAM untuk menjelaskan bagian gambar yang dilihat model
  - pemuatan model yang aman lintas versi Keras

Contoh:
    from inferensi import PrediktorSampah
    p = PrediktorSampah.dari_folder("model_final")
    hasil = p.prediksi("foto_botol.jpg")
    print(hasil["label"], hasil["keyakinan"], hasil["status"])
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_FINAL_DIR = BASE_DIR / "model_final"
IMAGE_SIZE = (224, 224)

STATUS_YAKIN = "yakin"
STATUS_TIDAK_YAKIN = "tidak_yakin"
STATUS_LUAR_DOMAIN = "di_luar_domain"

PESAN_STATUS = {
    STATUS_YAKIN: "Prediksi dapat dipercaya.",
    STATUS_TIDAK_YAKIN: "Model kurang yakin. Coba foto ulang dengan objek lebih jelas dan pencahayaan cukup.",
    STATUS_LUAR_DOMAIN: "Gambar tampaknya bukan foto sampah sehingga tidak diklasifikasikan.",
}


# ---------------------------------------------------------------------------
# Fungsi murni (mudah diuji dengan unit test)
# ---------------------------------------------------------------------------
def tentukan_status(keyakinan: float, skor_kemiripan: float | None,
                    ambang_keyakinan: float, ambang_ood: float | None) -> str:
    """Urutan keputusan: cek luar domain dulu, baru cek keyakinan."""
    if ambang_ood is not None and skor_kemiripan is not None and skor_kemiripan < ambang_ood:
        return STATUS_LUAR_DOMAIN
    if keyakinan < ambang_keyakinan:
        return STATUS_TIDAK_YAKIN
    return STATUS_YAKIN


def normalisasi_l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def hitung_centroid(fitur: np.ndarray, label: np.ndarray, jumlah_kelas: int) -> np.ndarray:
    """Rata-rata fitur ternormalisasi per kelas, lalu dinormalisasi lagi. Bentuk (kelas, dimensi)."""
    f = normalisasi_l2(fitur)
    return normalisasi_l2(np.stack([f[label == c].mean(axis=0) for c in range(jumlah_kelas)]))


def skor_kemiripan(fitur: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Kemiripan kosinus tertinggi terhadap centroid kelas mana pun (nilai -1..1)."""
    return (normalisasi_l2(fitur) @ centroid.T).max(axis=-1)


def energi_tepi(cam: np.ndarray, lebar: float = 0.15) -> float:
    """Proporsi 'perhatian' Grad-CAM yang jatuh di pita tepi gambar (proksi latar belakang)."""
    h, w = cam.shape
    bh, bw = int(h * lebar), int(w * lebar)
    total = float(cam.sum())
    if total <= 0:
        return float("nan")
    tengah = float(cam[bh:h - bh, bw:w - bw].sum())
    return (total - tengah) / total


# ---------------------------------------------------------------------------
# Preprocessing (identik dengan CNN_Sampah._load_image)
# ---------------------------------------------------------------------------
def muat_gambar(sumber) -> np.ndarray:
    """Terima path / PIL.Image / array uint8 HxWx3, kembalikan float32 224x224x3 rentang 0-1."""
    if isinstance(sumber, (str, Path)):
        image = tf.io.decode_image(tf.io.read_file(str(sumber)), channels=3, expand_animations=False)
    else:
        if isinstance(sumber, Image.Image):
            sumber = np.asarray(sumber.convert("RGB"))
        array = np.asarray(sumber)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:  # dianggap sudah 0-1 bila float
            array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
        image = tf.convert_to_tensor(array)
    image = tf.image.resize(image, IMAGE_SIZE)
    return (tf.cast(image, tf.float32) / 255.0).numpy()


# ---------------------------------------------------------------------------
# Pemuatan model yang aman lintas versi Keras
# ---------------------------------------------------------------------------
def _baca_config(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("config.json"))


def bangun_mobilenetv2(jumlah_kelas: int, dropout: float = 0.3) -> tf.keras.Model:
    """Struktur sama persis dengan eksperimen_perbaikan.build_transfer (tanpa unduh bobot)."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="ke_rentang_min1_1")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3))
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(jumlah_kelas, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="tl_mobilenetv2")


def muat_model_aman(path) -> tf.keras.Model:
    path = Path(path)
    try:
        return tf.keras.models.load_model(str(path), compile=False)
    except (TypeError, ValueError) as error:
        config = _baca_config(path)
        nama = config["config"]["name"]
        layers = config["config"]["layers"]
        kelas = next(l["config"]["units"] for l in layers if l["config"].get("name") == "class_probabilities")
        print(f"[INFO] Versi Keras berbeda ({str(error).splitlines()[0][:80]}...). "
              f"Membangun ulang arsitektur '{nama}' lalu memuat bobot.")
        if nama.startswith("tl_mobilenetv2"):
            ada_dropout = any(l["config"].get("name") == "head_dropout" for l in layers)
            rate = next((l["config"]["rate"] for l in layers if l["config"].get("name") == "head_dropout"), 0.0)
            model = bangun_mobilenetv2(kelas, rate if ada_dropout else 0.0)
        elif nama == "waste_classifier_cnn":
            import CNN_Sampah
            model = CNN_Sampah.build_cnn(kelas)
        elif nama == "cnn_sampah_stabil":
            from eksperimen_perbaikan import build_cnn_stabil
            model = build_cnn_stabil(kelas, 0.9)
        else:
            raise
        model.load_weights(str(path))
        return model


# ---------------------------------------------------------------------------
# Prediktor
# ---------------------------------------------------------------------------
class PrediktorSampah:
    def __init__(self, model_path, class_names: list[str], ambang_keyakinan: float = 0.0,
                 ambang_ood: float | None = None, centroid: np.ndarray | None = None):
        self.model = muat_model_aman(model_path)
        self.class_names = list(class_names)
        self.ambang_keyakinan = float(ambang_keyakinan)
        self.ambang_ood = None if ambang_ood is None else float(ambang_ood)
        self.centroid = centroid
        self._lapisan = [l for l in self.model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
        self._layer_conv = self._cari_layer_konvolusi()

    @classmethod
    def dari_folder(cls, folder=MODEL_FINAL_DIR) -> "PrediktorSampah":
        folder = Path(folder)
        cfg = json.loads((folder / "config_inferensi.json").read_text(encoding="utf-8"))
        centroid_path = folder / "centroid_fitur.npy"
        return cls(
            folder / cfg.get("file_model", "model_sampah.keras"),
            cfg["class_names"],
            ambang_keyakinan=cfg.get("ambang_keyakinan", 0.0),
            ambang_ood=cfg.get("ambang_ood"),
            centroid=np.load(centroid_path) if centroid_path.exists() else None,
        )

    # -- internal ---------------------------------------------------------
    def _cari_layer_konvolusi(self):
        for layer in reversed(self._lapisan):
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.MaxPooling2D)):
                continue
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except (AttributeError, ValueError):
                continue
        return None

    def _forward(self, batch):
        """Forward pass berurutan; mengembalikan (probabilitas, fitur GAP, output konvolusi)."""
        z, fitur, conv = batch, None, None
        for layer in self._lapisan:
            z = layer(z, training=False)
            if layer is self._layer_conv:
                conv = z
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                fitur = z
        return z, fitur, conv

    # -- API batch (untuk evaluasi) -----------------------------------------
    def prob_dan_fitur(self, images, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """images: array (N,224,224,3) 0-1, atau tf.data.Dataset berisi (gambar, label)."""
        probs, feats = [], []
        if isinstance(images, tf.data.Dataset):
            iterator = (x for x, *_ in images)
        else:
            iterator = (images[i:i + batch_size] for i in range(0, len(images), batch_size))
        for batch in iterator:
            p, f, _ = self._forward(tf.convert_to_tensor(batch, tf.float32))
            probs.append(np.asarray(p))
            feats.append(np.asarray(f))
        return np.concatenate(probs), np.concatenate(feats)

    # -- API satu gambar (untuk aplikasi) ----------------------------------
    def prediksi(self, sumber, top_k: int = 3) -> dict:
        image = muat_gambar(sumber)
        p, f, _ = self._forward(tf.convert_to_tensor(image[None], tf.float32))
        probs = np.asarray(p)[0]
        urut = np.argsort(-probs)[:top_k]
        kemiripan = None
        if self.centroid is not None:
            kemiripan = float(skor_kemiripan(np.asarray(f), self.centroid)[0])
        status = tentukan_status(float(probs[urut[0]]), kemiripan, self.ambang_keyakinan, self.ambang_ood)
        return {
            "label": self.class_names[urut[0]],
            "keyakinan": float(probs[urut[0]]),
            "top_k": [(self.class_names[i], float(probs[i])) for i in urut],
            "probabilitas": {n: float(v) for n, v in zip(self.class_names, probs)},
            "skor_kemiripan": kemiripan,
            "status": status,
            "pesan": PESAN_STATUS[status],
        }

    def gradcam(self, sumber, kelas: int | None = None) -> np.ndarray:
        """Peta Grad-CAM 224x224 bernilai 0-1 untuk kelas prediksi (atau kelas tertentu)."""
        image = sumber if isinstance(sumber, np.ndarray) and sumber.shape == (*IMAGE_SIZE, 3) \
            and sumber.dtype == np.float32 else muat_gambar(sumber)
        batch = tf.convert_to_tensor(image[None], tf.float32)
        with tf.GradientTape() as tape:
            probs, _, conv = self._forward(batch)
            if kelas is None:
                kelas = int(tf.argmax(probs[0]))
            skor = probs[:, kelas]
        grads = tape.gradient(skor, conv)
        bobot = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * bobot, axis=-1))
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return tf.image.resize(cam[..., None], IMAGE_SIZE).numpy()[..., 0]
