"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

FITUR UNIK + BUKTI DAMPAKNYA  (Tugas Anggota 4 - Bagian 3, untuk UTS)

Model final: A3_reg_saja (MobileNetV2 + regularisasi), dipilih berdasarkan
akurasi VALIDATION tertinggi (0.862) agar test set tidak ikut dipakai memilih model.

Fitur yang dibuktikan (sesuai format laporan bagian 5e):
  F1 Grad-CAM                       : perbandingan baseline vs model final + metrik energi tepi
  F2 Penanganan data tidak seimbang : class weight vs tanpa class weight (eksperimen A5)
  F3 Confidence threshold           : kurva cakupan vs akurasi, ambang dipilih dari data validation
  F4 Deteksi input di luar domain   : kemiripan kosinus ke centroid kelas, diuji dengan CIFAR-10
  F5 Kompresi model ke TFLite       : float32, dynamic-range int8, float16 (ukuran, akurasi, kecepatan)
  (+ unit test di test_sampah.py)

Output:
  model_final/        model + konfigurasi siap dipakai aplikasi (dipakai juga oleh unit test)
  hasil_fitur_unik/   grafik, tabel, dan ringkasan_fitur_unik.md untuk laporan

Cara pakai (dari folder yang sama dengan CNN_Sampah.py):
  py fitur_unik.py
  py fitur_unik.py --lewati-a5 --lewati-tflite     (bagian tertentu dilewati)
  py fitur_unik.py --ood-folder foto_bukan_sampah   (tambahan gambar luar domain milik sendiri)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

import CNN_Sampah as base
import inferensi as inf

BASE_DIR = Path(__file__).resolve().parent
HASIL_EKSPERIMEN = BASE_DIR / "hasil_eksperimen"
OUT = BASE_DIR / "hasil_fitur_unik"
MODEL_FINAL = BASE_DIR / "model_final"
DEFAULT_MODEL = HASIL_EKSPERIMEN / "A3_reg_saja" / "best_model.keras"
BASELINE_MODEL = BASE_DIR / "cnn_training_output" / "best_cnn_sampah.keras"
A5_MODEL = HASIL_EKSPERIMEN / "A5_reg_tanpa_class_weight" / "best_model.keras"
TARGET_AKURASI = 0.95       # target akurasi untuk prediksi yang diterima (F3)
TPR_OOD = 0.95              # 95% gambar sampah validation harus lolos filter luar domain (F4)

RINGKASAN: dict = {}


def judul(teks: str) -> None:
    print("\n" + "=" * 72 + f"\n{teks}\n" + "=" * 72)


def label_dari(samples):
    return np.array([l for _, l in samples])


# ---------------------------------------------------------------------------
# F3 Confidence threshold
# ---------------------------------------------------------------------------
def fitur_confidence_threshold(val, test):
    judul("F3. CONFIDENCE THRESHOLD")
    ambang = np.round(np.arange(0.30, 0.951, 0.05), 2)

    def kurva(probs, y):
        conf, benar = probs.max(1), probs.argmax(1) == y
        baris = []
        for t in ambang:
            m = conf >= t
            baris.append((t, m.mean(), benar[m].mean() if m.any() else np.nan))
        return baris

    k_val, k_test = kurva(*val), kurva(*test)
    kandidat = [t for t, cov, acc in k_val if not np.isnan(acc) and acc >= TARGET_AKURASI and cov >= 0.5]
    if kandidat:
        terpilih = float(kandidat[0])
        alasan = f"ambang terkecil dengan akurasi validation >= {TARGET_AKURASI:.0%} dan cakupan >= 50%"
    else:
        cukup = [t for t, cov, _ in k_val if cov >= 0.5]
        terpilih = float(cukup[-1]) if cukup else float(ambang[0])
        alasan = (f"target {TARGET_AKURASI:.0%} tidak tercapai; dipilih ambang tertinggi yang masih "
                  "menjawab >= 50% gambar validation")

    conf_t, benar_t = test[0].max(1), test[0].argmax(1) == test[1]
    m = conf_t >= terpilih
    hasil = {
        "ambang_terpilih": terpilih, "alasan": alasan,
        "akurasi_test_tanpa_ambang": float(benar_t.mean()),
        "akurasi_test_diterima": float(benar_t[m].mean()),
        "cakupan_test": float(m.mean()),
        "jumlah_ditolak_test": int((~m).sum()),
        "salah_yang_tertolak": int((~benar_t & ~m).sum()),
        "total_salah": int((~benar_t).sum()),
        "keyakinan_maks_test": float(conf_t.max()),
    }
    print(f"Ambang terpilih : {terpilih}  ({alasan})")
    print(f"Test tanpa ambang: akurasi {hasil['akurasi_test_tanpa_ambang']:.4f} (semua gambar dijawab)")
    print(f"Test dengan ambang: akurasi {hasil['akurasi_test_diterima']:.4f} pada {hasil['cakupan_test']:.1%} gambar; "
          f"{hasil['salah_yang_tertolak']} dari {hasil['total_salah']} prediksi salah berhasil ditahan")
    print("Catatan: label smoothing 0.1 membuat keyakinan maksimum model sekitar 0.91, "
          f"(maksimum di test: {hasil['keyakinan_maks_test']:.3f}).")

    with (OUT / "F3_tabel_ambang.csv").open("w", encoding="utf-8") as f:
        f.write("ambang,cakupan_val,akurasi_val,cakupan_test,akurasi_test\n")
        for (t, cv, av), (_, ct, at) in zip(k_val, k_test):
            f.write(f"{t:.2f},{cv:.4f},{av:.4f},{ct:.4f},{at:.4f}\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nama, k in (("Validation", k_val), ("Test", k_test)):
        ax[0].plot([b[0] for b in k], [b[2] for b in k], marker="o", label=f"Akurasi ({nama})")
        ax[0].plot([b[0] for b in k], [b[1] for b in k], ls="--", marker=".", label=f"Cakupan ({nama})")
    ax[0].axvline(terpilih, color="gray", ls=":", label=f"Ambang terpilih = {terpilih}")
    ax[0].set_xlabel("Ambang keyakinan")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Akurasi prediksi yang diterima vs cakupan")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].hist(conf_t[benar_t], bins=25, alpha=0.7, label="Prediksi benar")
    ax[1].hist(conf_t[~benar_t], bins=25, alpha=0.7, label="Prediksi salah")
    ax[1].axvline(terpilih, color="gray", ls=":")
    ax[1].set_xlabel("Keyakinan (probabilitas tertinggi)")
    ax[1].set_title("Sebaran keyakinan pada test set")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "F3_confidence_threshold.png", dpi=130)
    plt.close(fig)
    RINGKASAN["F3_confidence_threshold"] = hasil
    return terpilih


# ---------------------------------------------------------------------------
# F4 Deteksi input di luar domain
# ---------------------------------------------------------------------------
def muat_ood(jumlah: int = 600, folder: Path | None = None):
    """CIFAR-10 (pesawat, mobil, hewan, kapal, truk) sebagai contoh gambar bukan sampah."""
    (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    nama_cifar = ["pesawat", "mobil", "burung", "kucing", "rusa", "anjing", "katak", "kuda", "kapal", "truk"]
    y = y.ravel()
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], jumlah // 10, replace=False) for c in range(10)])
    gambar = tf.image.resize(x[idx], inf.IMAGE_SIZE).numpy() / 255.0
    sumber = [f"CIFAR-10: {nama_cifar[c]}" for c in y[idx]]
    if folder and Path(folder).is_dir():
        files = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if files:
            gambar = np.concatenate([gambar, np.stack([inf.muat_gambar(p) for p in files])])
            sumber += [f"Folder: {p.name}" for p in files]
    return gambar.astype(np.float32), sumber


def fitur_ood(pred, feat_val, feat_test, prob_test, prob_val, ood_folder):
    judul("F4. DETEKSI INPUT DI LUAR DOMAIN")
    x_ood, sumber_ood = muat_ood(folder=ood_folder)
    prob_ood, feat_ood = pred.prob_dan_fitur(x_ood)

    s_val = inf.skor_kemiripan(feat_val, pred.centroid)
    s_test = inf.skor_kemiripan(feat_test, pred.centroid)
    s_ood = inf.skor_kemiripan(feat_ood, pred.centroid)
    ambang = float(np.quantile(s_val, 1 - TPR_OOD))

    y_bin = np.r_[np.ones(len(s_test)), np.zeros(len(s_ood))]
    auroc_cos = roc_auc_score(y_bin, np.r_[s_test, s_ood])
    auroc_msp = roc_auc_score(y_bin, np.r_[prob_test.max(1), prob_ood.max(1)])
    hasil = {
        "metode": "kemiripan kosinus fitur GAP terhadap centroid kelas (data training)",
        "data_ood": f"{len(x_ood)} gambar ({'CIFAR-10' if not ood_folder else 'CIFAR-10 + folder'})",
        "ambang_ood": ambang,
        "sampah_test_lolos": float((s_test >= ambang).mean()),
        "ood_tertolak": float((s_ood < ambang).mean()),
        "auroc_kemiripan_kosinus": float(auroc_cos),
        "auroc_keyakinan_softmax": float(auroc_msp),
        "ood_tanpa_filter_dijawab_yakin": float((prob_ood.max(1) >= RINGKASAN.get(
            "F3_confidence_threshold", {}).get("ambang_terpilih", 0.0)).mean()),
    }
    print(f"Ambang kemiripan            : {ambang:.4f} (95% gambar sampah validation lolos)")
    print(f"Gambar sampah test yang lolos: {hasil['sampah_test_lolos']:.1%}")
    print(f"Gambar luar domain ditolak   : {hasil['ood_tertolak']:.1%}")
    print(f"AUROC kemiripan kosinus      : {auroc_cos:.4f}   (pembanding: keyakinan softmax {auroc_msp:.4f})")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(s_test, bins=40, alpha=0.7, density=True, label="Sampah (test)")
    ax.hist(s_ood, bins=40, alpha=0.7, density=True, label="Bukan sampah (CIFAR-10)")
    ax.axvline(ambang, color="k", ls="--", label=f"Ambang = {ambang:.3f}")
    ax.set_xlabel("Kemiripan kosinus ke centroid kelas terdekat")
    ax.set_title(f"Deteksi luar domain (AUROC = {auroc_cos:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "F4_ood_histogram.png", dpi=130)
    plt.close(fig)

    # contoh: tanpa filter model tetap memaksa memberi label sampah
    idx = np.random.default_rng(1).choice(len(x_ood), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    for a, i in zip(axes.ravel(), idx):
        a.imshow(x_ood[i])
        a.axis("off")
        lab = pred.class_names[int(prob_ood[i].argmax())]
        ditolak = s_ood[i] < ambang
        a.set_title(f"{sumber_ood[i]}\nTanpa filter: {lab} ({prob_ood[i].max():.2f})\n"
                    f"Dengan filter: {'DITOLAK' if ditolak else 'lolos'} (skor {s_ood[i]:.2f})",
                    fontsize=8.5, color="green" if ditolak else "crimson")
    fig.suptitle("Contoh gambar bukan sampah", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "F4_contoh_ood.png", dpi=110)
    plt.close(fig)
    RINGKASAN["F4_deteksi_luar_domain"] = hasil
    return ambang


# ---------------------------------------------------------------------------
# F1 Grad-CAM: baseline vs model final
# ---------------------------------------------------------------------------
def fitur_gradcam(pred_final, test_samples, class_names):
    judul("F1. GRAD-CAM: BASELINE vs MODEL FINAL")
    model_list = [("Model final", pred_final)]
    if BASELINE_MODEL.is_file():
        model_list.insert(0, ("Baseline", inf.PrediktorSampah(BASELINE_MODEL, class_names)))
    else:
        print(f"Baseline tidak ditemukan di {BASELINE_MODEL}, hanya model final yang dianalisis.")

    gambar = [inf.muat_gambar(p) for p, _ in test_samples]
    hasil = {}
    for nama, p in model_list:
        e = [inf.energi_tepi(p.gradcam(g)) for g in gambar]
        hasil[nama] = float(np.nanmean(e))
        print(f"{nama:12s}: rata-rata energi Grad-CAM di tepi gambar = {hasil[nama]:.3f}")
    print("(Peta panas yang rata di seluruh gambar memberi nilai ~0.51. Makin kecil berarti "
          "perhatian makin terpusat ke tengah/objek, bukan latar.)")

    rng = np.random.default_rng(3)
    idx = rng.choice(len(test_samples), 8, replace=False)
    baris = 1 + len(model_list)
    fig, axes = plt.subplots(baris, len(idx), figsize=(2.3 * len(idx), 2.6 * baris), squeeze=False)
    for col, i in enumerate(idx):
        axes[0, col].imshow(gambar[i])
        axes[0, col].set_title(f"Asli: {class_names[test_samples[i][1]]}", fontsize=8)
        for r, (nama, p) in enumerate(model_list, start=1):
            hasil_p = p.prediksi(gambar[i])
            axes[r, col].imshow(gambar[i])
            axes[r, col].imshow(p.gradcam(gambar[i]), cmap="jet", alpha=0.45)
            benar = hasil_p["label"] == class_names[test_samples[i][1]]
            axes[r, col].set_title(f"{nama}: {hasil_p['label']}", fontsize=7.5,
                                   color="green" if benar else "crimson")
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Grad-CAM: area yang diperhatikan model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "F1_gradcam_perbandingan.png", dpi=120)
    plt.close(fig)
    RINGKASAN["F1_gradcam"] = {"energi_tepi_rata2": hasil, "nilai_acuan_merata": 0.51}


# ---------------------------------------------------------------------------
# F2 Penanganan data tidak seimbang
# ---------------------------------------------------------------------------
def siapkan_a5(lewati: bool) -> None:
    """Latih A5 = konfigurasi A3 tanpa class weight (dijalankan PALING AWAL, sebelum model lain dimuat,
    karena proses training memanggil clear_session)."""
    if A5_MODEL.is_file() or lewati:
        return
    judul("MELATIH A5: sama dengan A3, hanya TANPA class weight (± 2-5 menit dengan GPU)")
    import eksperimen_perbaikan as ex
    cfg = {**ex.EKSPERIMEN["A3_reg_saja"], "deskripsi": "Ablasi: MobileNetV2 beku, regularisasi, TANPA class weight"}
    asli = base.calculate_class_weights
    base.calculate_class_weights = lambda samples, n: {i: 1.0 for i in range(n)}
    try:
        ex.jalankan("A5_reg_tanpa_class_weight", cfg)
    finally:
        base.calculate_class_weights = asli
    tf.keras.backend.clear_session()


def fitur_class_weight(model_final_path, test_samples, class_names):
    judul("F2. PENANGANAN DATA TIDAK SEIMBANG (CLASS WEIGHT)")
    a5_path = A5_MODEL
    if not a5_path.is_file():
        print("Model A5 (tanpa class weight) tidak ada, F2 dilewati.")
        return
    train_s, _, _, _ = base.collect_splits(base.DATASET_DIR)
    jumlah = np.bincount(label_dari(train_s), minlength=len(class_names))
    bobot = base.calculate_class_weights(train_s, len(class_names))
    y = label_dari(test_samples)
    ds = base.make_dataset(test_samples)
    rekap = {}
    for nama, path in (("Dengan class weight (A3)", model_final_path), ("Tanpa class weight (A5)", a5_path)):
        probs, _ = inf.PrediktorSampah(path, class_names).prob_dan_fitur(ds)
        yp = probs.argmax(1)
        _, rec, f1, _ = precision_recall_fscore_support(y, yp, labels=range(len(class_names)), zero_division=0)
        rekap[nama] = {"akurasi": float((yp == y).mean()), "macro_f1": float(f1_score(y, yp, average="macro")),
                       "recall": rec, "f1": f1}
    a, b = rekap.values()
    minoritas = [class_names[i] for i in np.argsort(jumlah)[:3]]
    print(f"{'Kelas':22s}{'train':>6s}{'bobot':>7s}{'recall+CW':>11s}{'recall-CW':>11s}")
    for i, k in enumerate(class_names):
        print(f"{k:22s}{jumlah[i]:6d}{bobot[i]:7.2f}{a['recall'][i]:11.3f}{b['recall'][i]:11.3f}")
    for nama, r in rekap.items():
        print(f"{nama}: akurasi {r['akurasi']:.4f}, macro F1 {r['macro_f1']:.4f}")

    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(x - 0.2, a["recall"], 0.4, label="Dengan class weight (A3)")
    ax.bar(x + 0.2, b["recall"], 0.4, label="Tanpa class weight (A5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={jumlah[i]})" for i, k in enumerate(class_names)], rotation=30,
                       ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (test)")
    ax.set_title("Pengaruh class weight terhadap recall per kelas")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "F2_class_weight_recall.png", dpi=130)
    plt.close(fig)
    RINGKASAN["F2_class_weight"] = {
        "bobot_kelas": {k: round(float(bobot[i]), 3) for i, k in enumerate(class_names)},
        "kelas_minoritas": minoritas,
        **{nama: {"akurasi": r["akurasi"], "macro_f1": r["macro_f1"],
                  "recall_minoritas": {k: float(r["recall"][class_names.index(k)]) for k in minoritas}}
           for nama, r in rekap.items()},
    }


# ---------------------------------------------------------------------------
# F5 Kompresi TFLite
# ---------------------------------------------------------------------------
def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def prediksi_tflite(path, images):
    it = _interpreter(path)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    hasil = []
    for img in images:
        it.set_tensor(inp["index"], img[None].astype(np.float32))
        it.invoke()
        hasil.append(it.get_tensor(out["index"])[0])
    return np.array(hasil)


def latensi_ms(fn, x, ulang=30):
    for _ in range(3):
        fn(x)
    t = time.perf_counter()
    for _ in range(ulang):
        fn(x)
    return (time.perf_counter() - t) / ulang * 1000


def fitur_tflite(pred, test_samples, prob_keras_test):
    judul("F5. KOMPRESI MODEL KE TFLITE")
    y = label_dari(test_samples)
    images = np.stack([inf.muat_gambar(p) for p, _ in test_samples]).astype(np.float32)
    keras_path = MODEL_FINAL / "model_sampah.keras"

    export_dir = Path(tempfile.mkdtemp()) / "saved_model"
    pred.model.export(str(export_dir))

    varian = {
        "TFLite float32": {},
        "TFLite float16": {"optimizations": [tf.lite.Optimize.DEFAULT], "types": [tf.float16]},
        "TFLite dynamic-range int8": {"optimizations": [tf.lite.Optimize.DEFAULT]},
    }
    tabel = []
    x1 = images[:1]
    with tf.device("/CPU:0"):
        ms_keras = latensi_ms(lambda x: pred.model(x, training=False), tf.convert_to_tensor(x1))
    tabel.append({"format": "Keras (.keras)", "ukuran_mb": keras_path.stat().st_size / 1e6,
                  "akurasi_test": float((prob_keras_test.argmax(1) == y).mean()),
                  "kesesuaian_dengan_keras": 1.0, "latensi_cpu_ms": ms_keras, "file": keras_path.name})

    for nama, opsi in varian.items():
        conv = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
        if "optimizations" in opsi:
            conv.optimizations = opsi["optimizations"]
        if "types" in opsi:
            conv.target_spec.supported_types = opsi["types"]
        file = MODEL_FINAL / ("model_sampah_" + nama.split()[1].replace("-", "_") + ".tflite")
        file.write_bytes(conv.convert())
        p = prediksi_tflite(file, images)
        it = _interpreter(file)
        inp, out = it.get_input_details()[0], it.get_output_details()[0]

        def jalan(x, it=it, inp=inp):
            it.set_tensor(inp["index"], x)
            it.invoke()

        tabel.append({"format": nama, "ukuran_mb": file.stat().st_size / 1e6,
                      "akurasi_test": float((p.argmax(1) == y).mean()),
                      "kesesuaian_dengan_keras": float((p.argmax(1) == prob_keras_test.argmax(1)).mean()),
                      "latensi_cpu_ms": latensi_ms(jalan, x1), "file": file.name})
    shutil.rmtree(export_dir.parent, ignore_errors=True)

    print(f"{'Format':28s}{'Ukuran MB':>10s}{'Akurasi':>9s}{'Sama dgn Keras':>16s}{'ms/gambar':>11s}")
    for r in tabel:
        print(f"{r['format']:28s}{r['ukuran_mb']:10.2f}{r['akurasi_test']:9.4f}"
              f"{r['kesesuaian_dengan_keras']:16.1%}{r['latensi_cpu_ms']:11.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    nama = [r["format"].replace("TFLite ", "TFLite\n") for r in tabel]
    b = ax[0].bar(nama, [r["ukuran_mb"] for r in tabel], color="#4C72B0")
    ax[0].bar_label(b, fmt="%.2f MB", fontsize=8)
    ax[0].set_title("Ukuran file model")
    b = ax[1].bar(nama, [r["akurasi_test"] for r in tabel], color="#55A868")
    ax[1].bar_label(b, fmt="%.4f", fontsize=8)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Akurasi test")
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "F5_tflite.png", dpi=130)
    plt.close(fig)
    RINGKASAN["F5_tflite"] = tabel


# ---------------------------------------------------------------------------
# Laporan ringkas
# ---------------------------------------------------------------------------
def tulis_markdown():
    r = RINGKASAN
    md = ["# Ringkasan Fitur Unik (model final: A3_reg_saja)", ""]
    if "F1_gradcam" in r:
        e = r["F1_gradcam"]["energi_tepi_rata2"]
        md += ["## F1 Grad-CAM", "Rata-rata proporsi perhatian di tepi gambar (acuan merata = 0.51):", ""]
        md += [f"- {k}: {v:.3f}" for k, v in e.items()] + [""]
    if "F2_class_weight" in r:
        f = r["F2_class_weight"]
        md += ["## F2 Class weight", "| Skenario | Akurasi | Macro F1 | " +
               " | ".join(f"Recall {k}" for k in f["kelas_minoritas"]) + " |",
               "|---|---|---|" + "---|" * len(f["kelas_minoritas"])]
        for nama in ("Dengan class weight (A3)", "Tanpa class weight (A5)"):
            v = f[nama]
            md.append(f"| {nama} | {v['akurasi']:.4f} | {v['macro_f1']:.4f} | " +
                      " | ".join(f"{v['recall_minoritas'][k]:.3f}" for k in f["kelas_minoritas"]) + " |")
        md.append("")
    if "F3_confidence_threshold" in r:
        f = r["F3_confidence_threshold"]
        md += ["## F3 Confidence threshold",
               f"- Ambang: **{f['ambang_terpilih']}** ({f['alasan']})",
               f"- Akurasi test tanpa ambang: {f['akurasi_test_tanpa_ambang']:.4f}",
               f"- Akurasi prediksi yang diterima: {f['akurasi_test_diterima']:.4f} "
               f"(cakupan {f['cakupan_test']:.1%})",
               f"- Prediksi salah yang berhasil ditahan: {f['salah_yang_tertolak']} dari {f['total_salah']}", ""]
    if "F4_deteksi_luar_domain" in r:
        f = r["F4_deteksi_luar_domain"]
        md += ["## F4 Deteksi luar domain",
               f"- Metode: {f['metode']}; data uji: {f['data_ood']}",
               f"- Ambang kemiripan: {f['ambang_ood']:.4f}",
               f"- Gambar sampah test yang lolos: {f['sampah_test_lolos']:.1%}",
               f"- Gambar bukan sampah yang ditolak: {f['ood_tertolak']:.1%}",
               f"- AUROC: {f['auroc_kemiripan_kosinus']:.4f} (pembanding keyakinan softmax: "
               f"{f['auroc_keyakinan_softmax']:.4f})", ""]
    if "F5_tflite" in r:
        md += ["## F5 Kompresi TFLite", "| Format | Ukuran (MB) | Akurasi test | Sama dengan Keras | CPU ms/gambar |",
               "|---|---|---|---|---|"]
        md += [f"| {t['format']} | {t['ukuran_mb']:.2f} | {t['akurasi_test']:.4f} | "
               f"{t['kesesuaian_dengan_keras']:.1%} | {t['latensi_cpu_ms']:.1f} |" for t in r["F5_tflite"]]
    (OUT / "ringkasan_fitur_unik.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "ringkasan_fitur_unik.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n" + "\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Bukti dampak fitur unik")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ood-folder", type=Path, default=None)
    parser.add_argument("--lewati-a5", action="store_true")
    parser.add_argument("--lewati-tflite", action="store_true")
    parser.add_argument("--lewati-gradcam", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(base.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_FINAL.mkdir(parents=True, exist_ok=True)
    if not args.model.is_file():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")

    siapkan_a5(args.lewati_a5)
    train_s, val_s, test_s, class_names = base.collect_splits(base.DATASET_DIR)
    shutil.copy(args.model, MODEL_FINAL / "model_sampah.keras")
    pred = inf.PrediktorSampah(MODEL_FINAL / "model_sampah.keras", class_names)

    judul("MENGHITUNG PROBABILITAS & FITUR (train / validation / test)")
    _, feat_train = pred.prob_dan_fitur(base.make_dataset(train_s))
    prob_val, feat_val = pred.prob_dan_fitur(base.make_dataset(val_s))
    prob_test, feat_test = pred.prob_dan_fitur(base.make_dataset(test_s))
    pred.centroid = inf.hitung_centroid(feat_train, label_dari(train_s), len(class_names))
    np.save(MODEL_FINAL / "centroid_fitur.npy", pred.centroid)
    print(f"Akurasi test model final: {(prob_test.argmax(1) == label_dari(test_s)).mean():.4f}")

    ambang_conf = fitur_confidence_threshold((prob_val, label_dari(val_s)), (prob_test, label_dari(test_s)))
    ambang_ood = fitur_ood(pred, feat_val, feat_test, prob_test, prob_val, args.ood_folder)

    metrik_final = {}
    m_path = args.model.parent / "metrics.json"
    if m_path.is_file():
        m = json.loads(m_path.read_text(encoding="utf-8"))
        metrik_final = {k: m[k] for k in ("test_accuracy", "test_macro_f1", "val_accuracy") if k in m}
    config = {
        "file_model": "model_sampah.keras", "sumber_model": str(args.model.parent.name),
        "class_names": class_names, "image_size": list(inf.IMAGE_SIZE),
        "ambang_keyakinan": ambang_conf, "ambang_ood": ambang_ood, "metrik": metrik_final,
    }
    (MODEL_FINAL / "config_inferensi.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    pred.ambang_keyakinan, pred.ambang_ood = ambang_conf, ambang_ood
    print(f"\nKonfigurasi inferensi disimpan: {MODEL_FINAL / 'config_inferensi.json'}")

    if not args.lewati_gradcam:
        fitur_gradcam(pred, test_s, class_names)
    fitur_class_weight(MODEL_FINAL / "model_sampah.keras", test_s, class_names)
    if not args.lewati_tflite:
        try:
            fitur_tflite(pred, test_s, prob_test)
        except Exception as error:
            print(f"F5 TFLite gagal: {error}")
    tulis_markdown()
    print(f"\nSemua hasil: {OUT}\nModel siap pakai: {MODEL_FINAL}")


if __name__ == "__main__":
    main()
