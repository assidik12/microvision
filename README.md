# MicroVision-Attend 👁️‍🗨️

> **Cost-effective local face attendance tracker powered by resource-efficient Edge AI computer vision.**

## 📌 Tentang Proyek

`microvision-attend` adalah sistem absensi berbasis _face recognition_ yang dioptimalkan khusus untuk berjalan di perangkat _edge_ dengan sumber daya komputasi minimal (_low-resource_). Proyek ini merupakan evolusi dan adaptasi modular dari penelitian sebelumnya yang berfokus pada efisiensi latensi, minimisasi penggunaan memori RAM, serta kemandirian komputasi lokal tanpa ketergantungan pada layanan _cloud_ berbiaya tinggi.

## 🚀 Fitur Utama

- **Edge-Optimized Pipeline**: Pemrosesan citra yang ringan agar dapat berjalan lancar di perangkat keras spesifikasi terbatas.
- **Local Biometric Authentication**: Ekstraksi fitur wajah secara lokal yang aman dan privasi terjamin.
- **Low-Resource Footprint**: Minimalisasi overhead memori menggunakan optimasi model inferensi.

## 🛠️ Instalasi & Menjalankan Proyek

1. Pastikan Anda telah mengaktifkan _virtual environment_ (`venv`).
2. Instal dependensi yang diperlukan:
   ```bash
   d:\project\microvision\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

> Catatan: environment pengembangan saat ini menggunakan Python 3.14 dan belum
> memiliki dependency proyek terpasang. Untuk deployment Raspberry Pi/Jetson,
> gunakan interpreter Python yang didukung wheel MediaPipe/TFLite pada perangkat
> tersebut, lalu jalankan perintah instalasi di atas.

## Menjalankan Aplikasi Absensi

Pastikan model `models/mobilefacenet.tflite` dan database identitas `data/embeddings.json` sudah tersedia, lalu jalankan:

```bash
python app.py --camera 0
```

Pipeline menggunakan:
1. **MediaPipe Face Detection (BlazeFace)**: Deteksi lokasi wajah secara real-time.
2. **MediaPipe Face Landmarker**: Penyelarasan orientasi mata (*5-point affine alignment*).
3. **MobileFaceNet (TFLite)**: Ekstraksi fitur embedding wajah berukuran 192 dimensi.
4. **Cosine Similarity Matching**: Pencocokan terhadap vektor referensi di `data/embeddings.json`.
5. **Attendance Recorder**: Otomatis mencatat kehadiran ke `data/attendance.csv` dengan cooldown anti-duplikasi.

---

## Registrasi Identitas Wajah Baru

Sistem tidak memerlukan proses pelatihan ulang (*zero-training*). Anda cukup mengekstrak embedding fitur wajah menggunakan script registrasi:

### 1. Registrasi Sekaligus dari Folder Dataset
Letakkan foto referensi dalam folder terstruktur:
```text
data/datasets/
  Budi/*.jpg
  Ani/*.jpg
```

Jalankan perintah registrasi batch:
```bash
python scripts/realtime_tflite_face_recognition.py \
  --register-dataset data/datasets \
  --registry data/embeddings.json \
  --confidence 0.5
```
> **Catatan Resiliency**: Script otomatis men-skip foto buram atau tidak ada wajah, dan mengambil rata-rata (*L2-normalized centroid*) dari foto valid.

### 2. Registrasi Satu Orang via Foto Referensi
```bash
python scripts/realtime_tflite_face_recognition.py \
  --register-name "Budi Santoso" \
  --reference-images data/datasets/Budi/1.jpg data/datasets/Budi/2.jpg
```

### 3. Pengujian Real-Time Langsung di Layar Webcam
```bash
python scripts/realtime_tflite_face_recognition.py --camera 0 --threshold 0.5
```

### 4. Visualisasi Matriks Kemiripan Embedding (Matplotlib)
```bash
python scripts/realtime_tflite_face_recognition.py --visualize
```

---

## 🔄 End-to-End Pipeline Scripts

Proyek ini menyediakan pipeline edge vision modular yang lengkap tanpa perlu melatih ulang model (*zero-training*):

### 1. Data Collection (`scripts/collect_dataset.py`)
Mengambil sampel wajah berkualitas tinggi secara interaktif dari webcam dengan validasi ukuran kotak wajah dan auto-align:
```bash
python scripts/collect_dataset.py "Ahmad Sofi Sidik" --count 15 --camera 0
```

### 2. Dataset Quality Audit & Validation (`scripts/validate_dataset.py`)
Memeriksa integritas gambar, deteksi wajah, duplicate hashing, dan kesiapan dataset sebelum registrasi:
```bash
python scripts/validate_dataset.py data/datasets --json data/validation_report.json
```

### 3. Verification Performance Benchmark & Evaluation (`scripts/evaluate_model.py`)
Mengukur performa pemisahan antar identitas (*intra vs inter class margin*), kurva trade-off FAR/FRR, dan menghasilkan chart evaluasi Matplotlib di `models/evaluation_report.png`:
```bash
python scripts/evaluate_model.py --dataset data/datasets --samples 8
```

