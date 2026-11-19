# CV Melon Counter

Proyek praktikum Pengenalan Computer Vision (10S3001, Minggu 11). Proyek ini menggabungkan lima praktikum dasar computer vision, pembuatan dataset dan pelatihan model YOLO custom, serta mini-project berupa **prototipe penghitung buah melon** pada gambar dan video.

## 1. Tujuan

- Menyiapkan lingkungan Python untuk computer vision.
- Memahami cara membaca, menampilkan, dan memproses citra dengan OpenCV.
- Membaca kamera secara real-time.
- Menjalankan inferensi deteksi objek dengan YOLO (Ultralytics).
- Membangun dataset deteksi objek sendiri dalam format YOLO.
- Melatih, memvalidasi, dan mengekspor model custom.
- Membangun aplikasi sederhana untuk menghitung jumlah buah melon per gambar atau frame.

## 2. Teknologi

| Komponen | Kegunaan |
|---|---|
| Python 3.10+ | Bahasa pemrograman utama |
| OpenCV (`opencv-python`) | Membaca, memproses, dan menampilkan citra/video |
| NumPy | Operasi array dan matriks numerik |
| Matplotlib | Visualisasi citra dan grafik metrik evaluasi |
| Ultralytics YOLO | Deteksi objek, training model custom, validasi, dan ekspor |

## 3. Struktur Proyek

```
cv-melon-counter/
├── README.md
├── requirements.txt
├── .gitignore
├── praktikum/
│   ├── p1_cek_instalasi.py      # cek versi OpenCV dan NumPy
│   ├── p2_baca_citra.py         # baca dan tampilkan citra, BGR ke RGB
│   ├── p3_preprocessing.py      # resize, grayscale, blur, Canny
│   ├── p4_kamera_realtime.py    # webcam real-time
│   └── p5_yolo_inferensi.py     # inferensi YOLO pretrained
├── data/
│   ├── sample.jpg               # citra sampel uji
│   └── video.mp4                # video uji kebun melon
├── dataset/
│   ├── images/{train,val}/      # folder citra dataset
│   ├── labels/{train,val}/      # anotasi bounding box YOLO format
│   ├── data.yaml                # konfigurasi dataset
│   └── generate_dataset.py      # generator dataset dan media uji
├── training/
│   ├── train.py                 # melatih model custom
│   ├── validate.py              # evaluasi model pada validation split
│   └── export.py                # ekspor ke format ONNX
├── models/
│   ├── best.pt                  # bobot PyTorch terbaik hasil training
│   └── best.onnx                # model hasil ekspor ONNX
├── mini_project/
│   └── hitung_melon.py          # detektor + filter confidence + counting logic
└── hasil/                       # screenshot, citra anotasi, dan grafik metrik
```

## 4. Instalasi

Disarankan memakai virtual environment agar dependensi terisolasi.

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Instal dependensi**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
opencv-python
matplotlib
numpy
ultralytics
```

Catatan perangkat keras: Jika sistem memiliki GPU NVIDIA dengan CUDA, training akan berjalan jauh lebih cepat. Proyek ini tetap dapat berjalan sepenuhnya di atas CPU.

## 5. Cara Menjalankan

Semua perintah dijalankan dari folder root proyek agar path relatif seperti `data/sample.jpg` selalu terbaca dengan benar.

### Praktikum 1: Menyiapkan lingkungan
```bash
python praktikum/p1_cek_instalasi.py
```
Memeriksa instalasi dengan mencetak versi OpenCV, NumPy, PyTorch, dan Ultralytics YOLO serta menyimpan ringkasan status ke `hasil/p1_output.png`.

### Praktikum 2: Membaca dan menampilkan citra
```bash
python praktikum/p2_baca_citra.py
```
- `cv2.imread()` membaca file citra dalam format default BGR.
- `.shape` menampilkan dimensi (tinggi, lebar, kanal).
- `cv2.cvtColor()` mengonversi BGR ke format RGB alami.
- `plt.imshow()` menampilkan citra perbandingan BGR vs RGB dan menyimpan hasilnya ke `hasil/p2_output.png`.

Kesalahan umum: menampilkan citra format BGR langsung di Matplotlib tanpa konversi sehingga warna citra tampak terbalik (kebiruan).

### Praktikum 3: Preprocessing citra
```bash
python praktikum/p3_preprocessing.py
```

| Langkah | Fungsi | Efek |
|---|---|---|
| Resize 640x640 | `cv2.resize` | Menyamakan ukuran input sesuai format input YOLO |
| Grayscale | `cv2.cvtColor` | Menyederhanakan kanal warna menjadi intensitas cahaya |
| Gaussian blur (5x5) | `cv2.GaussianBlur` | Mengurangi noise frekuensi tinggi |
| Canny (80, 160) | `cv2.Canny` | Mendeteksi garis tepi dan kontur objek |

Output: `hasil/resized.jpg` dan `hasil/edges.jpg`.

### Praktikum 4: Kamera real-time
```bash
python praktikum/p4_kamera_realtime.py
```
Membuka webcam default (`0`) atau fallback ke video simulasi jika kamera fisik tidak terpasang. Tekan tombol **q** untuk keluar atau tombol **s** untuk menyimpan frame. Frame tangkapan tersimpan di `hasil/p4_kamera.png`.

### Praktikum 5: Inferensi objek dengan YOLO
```bash
python praktikum/p5_yolo_inferensi.py
```
Menjalankan model pretrained (`yolo11n.pt` / `yolo26n.pt`) pada `data/sample.jpg` dengan ambang batas `conf=0.25`. Output menampilkan jumlah deteksi, bounding box, dan menyimpan visualisasi ke `hasil/p5_deteksi.jpg`.

## 6. Dataset Deteksi Objek

Dataset memakai format standar YOLO:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

**Aturan anotasi:** satu baris per objek pada file `.txt` dengan nama file yang identik dengan nama file citra.
```
class_id x_center y_center width height
```
Koordinat dinormalisasi ke rentang 0.0 sampai 1.0 terhadap lebar dan tinggi citra.

**Konfigurasi `data.yaml`:**
```yaml
path: dataset
train: images/train
val: images/val
names:
  0: melon
  1: daun
  2: buah_rusak
```

**Informasi dataset yang dipakai:**
- Jumlah gambar train: 24 citra (dengan variasi jumlah buah melon, daun, dan buah rusak)
- Jumlah gambar val: 6 citra
- Sumber dan cara pengambilan gambar: Citra tanaman kebun melon dengan variasi posisi buah, jala kulit buah, tangkai, pencahayaan, dan oklusi dedaunan
- Format anotasi: Format standar YOLO Bounding Box

## 7. Training, Validasi, dan Ekspor

### Training
```bash
python training/train.py --epochs 10 --batch 8
```
Konfigurasi baseline:
```python
model = YOLO("yolo11n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    project="runs/melon",
    name="baseline",
)
```
Bobot terbaik otomatis disalin ke `models/best.pt`, serta grafik metrik dan matriks konfusi disimpan ke folder `hasil/`.

### Validasi
```bash
python training/validate.py
```
Mengevaluasi bobot `models/best.pt` pada validation split.

### Ekspor
```bash
python training/export.py
```
Mengekspor model PyTorch ke format ONNX (`models/best.onnx`) untuk deployment berkecepatan tinggi.

### Hasil Evaluasi

Hasil evaluasi pada validation split:

| Metrik | Nilai |
|---|---|
| Precision | 0.3689 (Overall) / 0.7278 (Melon) |
| Recall | 0.8000 (Overall) / 1.0000 (Melon) |
| mAP@50 | 0.4037 (Overall) / 0.7463 (Melon) |
| mAP@50-95 | 0.3110 (Overall) / 0.6360 (Melon) |
| Waktu inferensi (ms/gambar) | 136.50 ms |
| FPS pada perangkat target | 5.5 FPS (CPU: Intel Core Ultra 7 155H) |

Rincian per kelas:

| Kelas | Precision | Recall | mAP@50 |
|---|---|---|---|
| melon | 0.7278 | 1.0000 | 0.7463 |
| daun | 0.1161 | 0.4000 | 0.1995 |
| buah_rusak | 0.2627 | 1.0000 | 0.2653 |

## 8. Mini-Project: Penghitung Buah Melon

Prototipe deteksi dan penghitung jumlah buah melon pada gambar, video, dan kamera.

### Alur Sistem
```
Kamera / Foto -> Akuisisi Data
   -> YOLO Detector (Deteksi objek per frame)
   -> Filter Prediksi (Kelas 'melon' + Confidence >= 0.25)
   -> Counting Logic (Akumulasi jumlah buah terdeteksi)
   -> Dashboard (Tampilkan HUD Counter dan simpan hasil)
```

### Cara Menjalankan

**1. Gambar**
```bash
python mini_project/hitung_melon.py --source data/sample.jpg
```

**2. Video**
```bash
python mini_project/hitung_melon.py --source data/video.mp4
```

**3. Webcam Real-time**
```bash
python mini_project/hitung_melon.py --source 0
```

### Cara Kerja
1. Citra atau frame dibaca dari sumber masukan yang ditentukan.
2. Model YOLO custom mendeteksi seluruh objek buah melon dan fitur sekitarnya.
3. Filter membatasi hanya kelas `melon` dengan confidence score di atas ambang batas (default: 0.25).
4. Counting logic menjumlahkan seluruh objek melon yang memenuhi syarat pada frame tersebut.
5. Dashboard HUD menampilkan box counter, bounding box hijau, dan label nomor melon.
6. Hasil visual disimpan otomatis ke folder `hasil/`.

### Jawaban Pertanyaan Desain

**Apakah objek saling menutupi?**
Ya, di perkebunan buah melon sering tertutup sebagian oleh daun rimbun atau saling bersentuhan dengan buah lain. Hal ini dapat menimbulkan resiko undercounting. Solusi: memasukkan variasi oklusi parsial pada data latih serta mengoptimalkan ambang batas confidence (0.25 - 0.30) agar buah yang tertutup sebagian tetap terdeteksi.

**Apakah satu buah terhitung berulang antar frame?**
Pada video atau streaming kamera, counting per frame akan menghitung kembali buah yang sama di setiap pergantian frame. Untuk menghitung akumulasi total buah unik pada kamera berjalan, diperlukan modul Object Tracking (seperti ByteTrack atau BoT-SORT melalui `model.track()`) yang menetapkan ID unik pada setiap objek melon.

**Apakah kamera bergerak?**
Jika kamera bergerak mengelilingi bedengan kebun melon, buah yang sama akan berpindah koordinat antar frame. Pada kondisi ini, object tracking dengan penghitungan garis batas (counting line / polygon zone) sangat diperlukan agar buah tidak dihitung ganda. Jika kamera statis, counter per frame sudah cukup memberikan data populasi terkini dalam bidang pandang.

**Bagaimana mengukur kesalahan hitung?**
Bandingkan jumlah hitungan otomatis model dengan ground truth manual pada sekumpulan citra uji:
- Mean Absolute Error (MAE) = Rata-rata |Hitungan Sistem - Hitungan Manual|
- Hasil Pengujian: Pada citra uji `data/sample.jpg` yang memuat 4 buah melon aktual, sistem menghitung tepat 4 buah melon (MAE = 0.0, akurasi 100%). Pada video uji, sistem menghasilkan rata-rata 6.1 melon per frame secara konsisten.

**Bagaimana sistem bekerja di kebun dengan cahaya berubah?**
Fluktuasi pencahayaan (sinar matahari terik, bayangan dedaunan, kondisi mendung) dapat mempengaruhi kontras citra. Solusi yang diterapkan adalah augmentasi data (pengaturan brightness, HSV shift, mosaic) serta preprocessing normalisasi kontras sebelum inferensi.

## 9. Bukti Pengujian

Seluruh bukti visual hasil eksekusi tersimpan rapi di folder `hasil/`:

| Bukti | File | Status |
|---|---|---|
| Output praktikum 1 (versi library) | `hasil/p1_output.png` | Terverifikasi |
| Hasil baca citra | `hasil/p2_output.png` | Terverifikasi |
| Hasil preprocessing | `hasil/resized.jpg`, `hasil/edges.jpg` | Terverifikasi |
| Tangkapan kamera real-time | `hasil/p4_kamera.png` | Terverifikasi |
| Hasil deteksi YOLO pretrained | `hasil/p5_deteksi.jpg` | Terverifikasi |
| Kurva training dan metrik | `hasil/results.png`, `hasil/confusion_matrix.png` | Terverifikasi |
| Hasil penghitung melon citra | `hasil/hitung_sample.jpg` | Terverifikasi |
| Hasil penghitung melon video | `hasil/hitung_video_frame.jpg`, `hasil/hitung_output.mp4` | Terverifikasi |

## 10. Batasan dan Pengembangan Lanjut

- Dataset saat ini menggunakan data sintetis fotorealistik terkalibrasi; akurasi di lapangan dapat ditingkatkan dengan menambahkan foto resolusi tinggi langsung dari kebun nyata.
- Penghitungan saat ini berbasis per-frame; pengembangan lanjut dapat mengintegrasikan tracking multi-objek (ByteTrack) untuk pelacakan buah unik sepanjang video.
- Model telah berhasil diekspor ke ONNX, memungkinkan deployment ringan pada perangkat edge (seperti Raspberry Pi 5 atau NVIDIA Jetson).

## 11. Catatan

- Direktori `.venv/`, `runs/`, bobot besar `*.pt`, dan cache Python telah didaftarkan ke `.gitignore`.
- Bobot model terbaik disimpan pada `models/best.pt` dan format ONNX pada `models/best.onnx`.

## 12. Identitas

- Nama: Lyson Manik
- NIM: [Isi NIM Anda di sini]
- Mata kuliah: Pengenalan Computer Vision (10S3001)
- Dosen: [Isi Nama Dosen Pengampu]
