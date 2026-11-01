"""
Penanggung jawab : Anggota 4 - [NAMA] ([NIM])

APLIKASI WEB KLASIFIKASI JENIS SAMPAH (Streamlit)

Memakai inferensi.PrediktorSampah dan folder model_final/ sehingga hasil di aplikasi
sama dengan angka di laporan. Fitur unik yang bisa didemokan:
  - Grad-CAM (area gambar yang diperhatikan model)
  - Confidence threshold (peringatan "kurang yakin")
  - Deteksi input di luar domain (menolak foto yang bukan sampah)
  - Kedua filter bisa dinyalakan/dimatikan di sidebar untuk membandingkan hasil

Menjalankan secara lokal:  streamlit run streamlit_app.py
Deployment: Streamlit Community Cloud (lihat PANDUAN_DEPLOYMENT.md)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Klasifikasi Jenis Sampah", page_icon="♻️", layout="wide")

import inferensi as inf  # noqa: E402  (diimpor setelah set_page_config)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model_final"
CONTOH_DIR = BASE / "contoh"

KELAS_DEFAULT = ["cardboard", "Food Organics", "glass", "metal", "Miscellaneous Trash",
                 "paper", "Plastic", "Textile Trash", "trash", "Vegetation"]

INFO_KELAS = {
    "cardboard": ("Kardus", "Daur ulang. Lipat agar ringkas dan pastikan kering serta tidak berminyak."),
    "Food Organics": ("Sisa makanan", "Sampah organik. Bisa dijadikan kompos; pisahkan dari plastik pembungkus."),
    "glass": ("Kaca", "Daur ulang. Bilas dan hati-hati dengan pecahan; bungkus kaca pecah sebelum dibuang."),
    "metal": ("Logam / kaleng", "Daur ulang. Bilas kaleng dari sisa isi sebelum dibuang ke tempat sampah anorganik."),
    "Miscellaneous Trash": ("Sampah campuran", "Residu. Buang ke tempat sampah umum jika tidak bisa dipilah lagi."),
    "paper": ("Kertas", "Daur ulang. Jaga tetap kering; kertas berminyak atau basah sebaiknya ke residu."),
    "Plastic": ("Plastik", "Daur ulang. Bilas dan keringkan; pisahkan tutup botol bila memungkinkan."),
    "Textile Trash": ("Tekstil / kain", "Pakaian layak dapat didonasikan; kain rusak dapat dijadikan lap atau dibuang ke residu."),
    "trash": ("Sampah umum", "Residu. Buang ke tempat sampah umum."),
    "Vegetation": ("Daun / tanaman", "Sampah organik. Cocok untuk kompos."),
}


@st.cache_resource(show_spinner="Memuat model (hanya sekali)...")
def muat_prediktor():
    cfg_path = MODEL_DIR / "config_inferensi.json"
    if cfg_path.exists():
        return inf.PrediktorSampah.dari_folder(MODEL_DIR), json.loads(cfg_path.read_text(encoding="utf-8"))
    return inf.PrediktorSampah(MODEL_DIR / "model_sampah.keras", KELAS_DEFAULT), {}


def overlay_gradcam(gambar: np.ndarray, cam: np.ndarray) -> Image.Image:
    campur = np.clip(0.55 * gambar + 0.45 * cm.jet(cam)[..., :3], 0, 1)
    return Image.fromarray((campur * 255).astype(np.uint8))


PRED, CFG = muat_prediktor()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Fitur")
    pakai_ambang = st.checkbox("Confidence threshold", value=True,
                               help=f"Prediksi dengan keyakinan di bawah {PRED.ambang_keyakinan:.0%} "
                                    "ditandai 'kurang yakin'.")
    pakai_ood = st.checkbox("Deteksi bukan sampah", value=PRED.ambang_ood is not None,
                            disabled=PRED.ambang_ood is None,
                            help="Menolak foto yang tidak mirip foto sampah mana pun.")
    st.divider()
    st.subheader("Tentang model")
    st.markdown("MobileNetV2 (transfer learning) + regularisasi, 10 kelas, input 224x224.")
    m = CFG.get("metrik", {})
    if m:
        st.markdown(f"**Akurasi test:** {m.get('test_accuracy', 0):.1%}  \n"
                    f"**Macro F1:** {m.get('test_macro_f1', 0):.3f} (516 gambar test)")
    st.caption("Kelas: " + ", ".join(PRED.class_names))

# --------------------------------------------------------------------------- input
st.title("♻️ Klasifikasi Jenis Sampah Berbasis Citra")
st.write("Upload atau foto sampah, lalu model CNN akan menebak jenisnya dan memberi saran pembuangan.")

contoh = sorted(p for p in CONTOH_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) \
    if CONTOH_DIR.is_dir() else []
pilihan_sumber = ["Upload foto", "Kamera"] + (["Contoh gambar"] if contoh else [])
sumber = st.radio("Sumber gambar", pilihan_sumber, horizontal=True)

foto = None
if sumber == "Upload foto":
    berkas = st.file_uploader("Pilih foto", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if berkas is not None:
        foto = Image.open(berkas)
elif sumber == "Kamera":
    berkas = st.camera_input("Ambil foto")
    if berkas is not None:
        foto = Image.open(berkas)
else:
    nama = st.selectbox("Pilih contoh (dari test set)", [p.stem.replace("_", " ") for p in contoh])
    foto = Image.open(contoh[[p.stem.replace("_", " ") for p in contoh].index(nama)])

if foto is None:
    st.info("Silakan pilih atau ambil foto sampah terlebih dahulu.")
    st.stop()

foto = ImageOps.exif_transpose(foto).convert("RGB")   # foto HP sering berputar karena EXIF

# --------------------------------------------------------------------------- prediksi
with st.spinner("Memproses..."):
    gambar = inf.muat_gambar(foto)
    hasil = PRED.prediksi(foto)
    cam = PRED.gradcam(gambar)

ambang = PRED.ambang_keyakinan if pakai_ambang else 0.0
ambang_ood = PRED.ambang_ood if pakai_ood else None
status = inf.tentukan_status(hasil["keyakinan"], hasil["skor_kemiripan"], ambang, ambang_ood)
label = hasil["label"]
nama_id, saran = INFO_KELAS.get(label, (label, "-"))

kiri, kanan = st.columns(2)
with kiri:
    st.image(foto, caption="Foto yang diklasifikasikan", width=360)
    st.image(overlay_gradcam(gambar, cam), caption="Grad-CAM: area yang diperhatikan model", width=360)
with kanan:
    if status == inf.STATUS_LUAR_DOMAIN:
        st.error(f"### 🚫 Bukan foto sampah\n{inf.PESAN_STATUS[status]}\n\n"
                 f"*(Tanpa filter, model akan menebak **{label}** dengan keyakinan {hasil['keyakinan']:.1%}.)*")
    elif status == inf.STATUS_TIDAK_YAKIN:
        st.warning(f"### ⚠️ Kurang yakin: kemungkinan **{label}** ({nama_id})\n"
                   f"Keyakinan {hasil['keyakinan']:.1%} di bawah ambang {ambang:.0%}.\n\n"
                   f"{inf.PESAN_STATUS[status]}")
    else:
        st.success(f"### ✅ {label} ({nama_id})\nKeyakinan: **{hasil['keyakinan']:.1%}**")
        st.markdown(f"**Saran pembuangan:** {saran}")

    st.subheader("Top-3 prediksi")
    for nama_kelas, p in hasil["top_k"]:
        st.progress(float(p), text=f"{nama_kelas}: {p:.1%}")

    with st.expander("Detail teknis"):
        st.write(f"- Ambang keyakinan: {'%.2f' % ambang if pakai_ambang else 'nonaktif'}")
        if hasil["skor_kemiripan"] is not None:
            st.write(f"- Skor kemiripan ke foto sampah: {hasil['skor_kemiripan']:.3f}"
                     + (f" (ambang {ambang_ood:.3f})" if ambang_ood is not None else " (filter nonaktif)"))
        st.write("- Probabilitas semua kelas:")
        st.dataframe({"kelas": list(hasil["probabilitas"].keys()),
                      "probabilitas": [round(v, 4) for v in hasil["probabilitas"].values()]},
                     hide_index=True)
