| Eksperimen | Akurasi test | Macro F1 | Top-3 | Δ akurasi vs baseline | Keterangan |
|---|---|---|---|---|---|
| A1_tanpa_aug_tanpa_reg | 0.8391 | 0.8243 | 0.9787 | +28.5 poin | Ablasi: MobileNetV2 beku, tanpa augmentasi, tanpa regularisasi |
| A2_aug_saja | 0.8062 | 0.7777 | 0.9767 | +25.2 poin | Ablasi: MobileNetV2 beku, augmentasi saja |
| A3_reg_saja | 0.8333 | 0.8175 | 0.9806 | +27.9 poin | Ablasi: MobileNetV2 beku, regularisasi saja |
| A4_aug_dan_reg | 0.8159 | 0.7919 | 0.9690 | +26.2 poin | Ablasi: MobileNetV2 beku, augmentasi + regularisasi |
| A5_reg_tanpa_class_weight | 0.8256 | 0.8029 | 0.9748 | +27.1 poin | Ablasi: MobileNetV2 beku, regularisasi, TANPA class weight |
| E0_baseline | 0.5543 | 0.5411 | 0.8818 | +0.0 poin | CNN 4 blok dari nol (model Anggota 3) |
| E1_cnn_stabil | 0.6609 | 0.6392 | 0.8779 | +10.7 poin | CNN baseline + AdamW, warmup-cosine LR 5e-4, BN momentum 0.9, label smoothing 0.1, augmentasi kuat |
| E2_mobilenetv2 | 0.8275 | 0.8111 | 0.9748 | +27.3 poin | Transfer learning MobileNetV2 + augmentasi + regularisasi, fine-tuning 40 layer |