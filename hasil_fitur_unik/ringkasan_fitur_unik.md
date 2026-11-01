# Ringkasan Fitur Unik (model final: A3_reg_saja)

## F1 Grad-CAM
Rata-rata proporsi perhatian di tepi gambar (acuan merata = 0.51):

- Baseline: 0.452
- Model final: 0.261

## F2 Class weight
| Skenario | Akurasi | Macro F1 | Recall trash | Recall Textile Trash | Recall Food Organics |
|---|---|---|---|---|---|
| Dengan class weight (A3) | 0.8333 | 0.8175 | 1.000 | 0.758 | 0.881 |
| Tanpa class weight (A5) | 0.8256 | 0.8029 | 0.733 | 0.758 | 0.881 |

## F3 Confidence threshold
- Ambang: **0.55** (ambang terkecil dengan akurasi validation >= 95% dan cakupan >= 50%)
- Akurasi test tanpa ambang: 0.8333
- Akurasi prediksi yang diterima: 0.8993 (cakupan 78.9%)
- Prediksi salah yang berhasil ditahan: 45 dari 86

## F4 Deteksi luar domain
- Metode: kemiripan kosinus fitur GAP terhadap centroid kelas (data training); data uji: 600 gambar (CIFAR-10)
- Ambang kemiripan: 0.5768
- Gambar sampah test yang lolos: 94.0%
- Gambar bukan sampah yang ditolak: 96.0%
- AUROC: 0.9861 (pembanding keyakinan softmax: 0.8761)

## F5 Kompresi TFLite
| Format | Ukuran (MB) | Akurasi test | Sama dengan Keras | CPU ms/gambar |
|---|---|---|---|---|
| Keras (.keras) | 9.78 | 0.8333 | 100.0% | 255.3 |
| TFLite float32 | 8.94 | 0.8333 | 100.0% | 8.9 |
| TFLite float16 | 4.53 | 0.8333 | 100.0% | 9.2 |
| TFLite dynamic-range int8 | 2.54 | 0.8256 | 97.5% | 22.3 |