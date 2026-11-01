"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

UNIT TEST PROYEK KLASIFIKASI SAMPAH  (pytest)

Menguji bagian-bagian penting pipeline agar hasil di laporan, kode, dan aplikasi
konsisten dan bebas kebocoran data.

Cara menjalankan (dari folder yang sama dengan CNN_Sampah.py):
  py -m pytest test_sampah.py -v
  (Colab: !python -m pytest test_sampah.py -v)

Test yang membutuhkan dataset atau model_final otomatis di-skip jika file belum ada.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import CNN_Sampah as base
import inferensi as inf

MODEL_FINAL = Path(__file__).resolve().parent / "model_final"
ADA_DATASET = base.DATASET_DIR.is_dir()
ADA_MODEL = (MODEL_FINAL / "config_inferensi.json").is_file()

perlu_dataset = pytest.mark.skipif(not ADA_DATASET, reason="folder datasett_preprocessed tidak ada")
perlu_model = pytest.mark.skipif(not ADA_MODEL, reason="model_final belum dibuat (jalankan fitur_unik.py)")

# jumlah dari log training Anggota 3 (train, val, test)
SPLIT_BASELINE = {
    "cardboard": (398, 49, 51), "Food Organics": (328, 41, 42), "glass": (468, 58, 60),
    "metal": (453, 56, 58), "Miscellaneous Trash": (396, 49, 50), "paper": (555, 69, 70),
    "Plastic": (736, 92, 93), "Textile Trash": (254, 31, 33), "trash": (109, 13, 15),
    "Vegetation": (347, 43, 44),
}


@pytest.fixture(scope="module")
def splits():
    return base.collect_splits(base.DATASET_DIR)


@pytest.fixture(scope="module")
def prediktor():
    return inf.PrediktorSampah.dari_folder(MODEL_FINAL)


def gambar_acak(lebar=500, tinggi=300, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (tinggi, lebar, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 1. Pembagian data
# ---------------------------------------------------------------------------
@perlu_dataset
def test_split_tidak_ada_kebocoran(splits):
    train, val, test, _ = splits
    p_train, p_val, p_test = ({p for p, _ in s} for s in (train, val, test))
    assert not p_train & p_val, "Ada gambar yang sama di train dan validation"
    assert not p_train & p_test, "Ada gambar yang sama di train dan test"
    assert not p_val & p_test, "Ada gambar yang sama di validation dan test"


@perlu_dataset
def test_split_reprodusibel(splits):
    ulang = base.collect_splits(base.DATASET_DIR)
    assert ulang == splits, "Pembagian data berubah antar pemanggilan (seed tidak konsisten)"


@perlu_dataset
def test_split_sama_dengan_baseline(splits):
    train, val, test, kelas = splits
    for i, k in enumerate(kelas):
        n = tuple(sum(l == i for _, l in s) for s in (train, val, test))
        assert n == SPLIT_BASELINE[k], f"Split kelas {k} berbeda dari training baseline: {n}"


@perlu_dataset
def test_proporsi_split_80_10_10(splits):
    train, val, test, _ = splits
    total = len(train) + len(val) + len(test)
    assert abs(len(train) / total - 0.8) < 0.01
    assert abs(len(val) / total - 0.1) < 0.01
    assert abs(len(test) / total - 0.1) < 0.01


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def test_preprocess_bentuk_dan_rentang():
    x = inf.muat_gambar(gambar_acak())
    assert x.shape == (224, 224, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_preprocess_menerima_berbagai_input(tmp_path):
    img = gambar_acak()
    path = tmp_path / "contoh.png"
    img.save(path)
    rgba = img.convert("RGBA")
    abu = img.convert("L")
    for sumber in (img, path, str(path), np.asarray(img), rgba, np.asarray(abu)):
        assert inf.muat_gambar(sumber).shape == (224, 224, 3)


def test_preprocess_identik_dengan_training(tmp_path):
    """Aplikasi harus memproses gambar PERSIS sama dengan pipeline training."""
    import tensorflow as tf
    path = tmp_path / "contoh.png"
    gambar_acak(640, 480).save(path)
    x_train, _ = base._load_image(tf.constant(str(path)), tf.constant(0))
    np.testing.assert_allclose(inf.muat_gambar(path), x_train.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Class weight (penanganan data tidak seimbang)
# ---------------------------------------------------------------------------
def test_class_weight_kelas_minoritas_terbesar():
    samples = [("a", 0)] * 100 + [("b", 1)] * 50 + [("c", 2)] * 10
    w = base.calculate_class_weights(samples, 3)
    assert w[2] > w[1] > w[0]
    # total kontribusi tiap kelas menjadi setara: jumlah * bobot = konstan
    assert np.allclose([100 * w[0], 50 * w[1], 10 * w[2]], 160 / 3)


# ---------------------------------------------------------------------------
# 4. Logika confidence threshold & deteksi luar domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("conf, sim, harap", [
    (0.90, 0.80, inf.STATUS_YAKIN),
    (0.40, 0.80, inf.STATUS_TIDAK_YAKIN),
    (0.90, 0.10, inf.STATUS_LUAR_DOMAIN),   # luar domain diprioritaskan walau keyakinan tinggi
    (0.40, 0.10, inf.STATUS_LUAR_DOMAIN),
    (0.70, 0.50, inf.STATUS_YAKIN),          # tepat di ambang = diterima
])
def test_tentukan_status(conf, sim, harap):
    assert inf.tentukan_status(conf, sim, ambang_keyakinan=0.70, ambang_ood=0.50) == harap


def test_tentukan_status_tanpa_filter_ood():
    assert inf.tentukan_status(0.9, None, 0.7, None) == inf.STATUS_YAKIN


def test_skor_kemiripan_dan_centroid():
    fitur = np.array([[1, 0], [2, 0], [0, 1], [0, 3]], dtype=np.float32)
    label = np.array([0, 0, 1, 1])
    c = inf.hitung_centroid(fitur, label, 2)
    np.testing.assert_allclose(c, [[1, 0], [0, 1]], atol=1e-6)
    s = inf.skor_kemiripan(np.array([[5, 0], [1, 1]], dtype=np.float32), c)
    np.testing.assert_allclose(s, [1.0, np.sqrt(0.5)], atol=1e-6)


def test_energi_tepi():
    rata = np.ones((100, 100))
    assert abs(inf.energi_tepi(rata) - (1 - 0.7 ** 2)) < 1e-6
    tengah = np.zeros((100, 100))
    tengah[40:60, 40:60] = 1
    assert inf.energi_tepi(tengah) == 0.0


# ---------------------------------------------------------------------------
# 5. Model final
# ---------------------------------------------------------------------------
@perlu_model
def test_config_inferensi_lengkap():
    cfg = json.loads((MODEL_FINAL / "config_inferensi.json").read_text(encoding="utf-8"))
    for kunci in ("class_names", "ambang_keyakinan", "ambang_ood", "file_model"):
        assert kunci in cfg
    assert len(cfg["class_names"]) == 10
    assert 0 < cfg["ambang_keyakinan"] < 1


@perlu_model
def test_output_prediksi_valid(prediktor):
    hasil = prediktor.prediksi(gambar_acak())
    p = np.array(list(hasil["probabilitas"].values()))
    assert len(p) == 10
    assert abs(p.sum() - 1.0) < 1e-4
    assert hasil["label"] in prediktor.class_names
    assert hasil["status"] in (inf.STATUS_YAKIN, inf.STATUS_TIDAK_YAKIN, inf.STATUS_LUAR_DOMAIN)
    prob_top = [v for _, v in hasil["top_k"]]
    assert prob_top == sorted(prob_top, reverse=True) and len(prob_top) == 3


@perlu_model
def test_prediksi_deterministik(prediktor):
    img = gambar_acak(seed=5)
    a = np.array(list(prediktor.prediksi(img)["probabilitas"].values()))
    b = np.array(list(prediktor.prediksi(img)["probabilitas"].values()))
    np.testing.assert_allclose(a, b, atol=1e-5)


@perlu_model
def test_gradcam_bentuk_dan_rentang(prediktor):
    cam = prediktor.gradcam(gambar_acak())
    assert cam.shape == (224, 224)
    assert cam.min() >= 0 and cam.max() <= 1.0 + 1e-6


@perlu_model
@perlu_dataset
def test_akurasi_model_final_sesuai_laporan(prediktor, splits):
    """Angka akurasi di aplikasi harus sama dengan angka di laporan (metrics.json)."""
    cfg = json.loads((MODEL_FINAL / "config_inferensi.json").read_text(encoding="utf-8"))
    if "test_accuracy" not in cfg.get("metrik", {}):
        pytest.skip("metrik tidak tercatat di config")
    _, _, test, _ = splits
    probs, _ = prediktor.prob_dan_fitur(base.make_dataset(test))
    akurasi = (probs.argmax(1) == np.array([l for _, l in test])).mean()
    assert abs(akurasi - cfg["metrik"]["test_accuracy"]) < 0.005


@perlu_model
@perlu_dataset
def test_tflite_konsisten_dengan_keras(prediktor, splits):
    file = MODEL_FINAL / "model_sampah_float16.tflite"
    if not file.is_file():
        pytest.skip("file TFLite belum dibuat")
    from fitur_unik import prediksi_tflite
    _, _, test, _ = splits
    gambar = np.stack([inf.muat_gambar(p) for p, _ in test[::25]])
    p_tfl = prediksi_tflite(file, gambar)
    p_keras, _ = prediktor.prob_dan_fitur(gambar)
    assert (p_tfl.argmax(1) == p_keras.argmax(1)).mean() >= 0.95
