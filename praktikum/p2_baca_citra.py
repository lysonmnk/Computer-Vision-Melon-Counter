"""
Praktikum 2: Membaca dan menampilkan citra.
- cv2.imread() membaca file citra (default BGR).
- shape menampilkan dimensi (tinggi, lebar, kanal).
- cv2.cvtColor() mengonversi BGR ke RGB.
- plt.imshow() menampilkan visualisasi perbandingan BGR vs RGB.
- Menyimpan hasil ke hasil/p2_output.png.
"""

import os
import cv2
import matplotlib.pyplot as plt

def main():
    image_path = "data/sample.jpg"
    print("=" * 50)
    print("PRAKTIKUM 2: MEMBACA DAN MENAMPILKAN CITRA")
    print("=" * 50)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File citra tidak ditemukan pada path: {image_path}")

    # 1. Membaca citra dengan OpenCV (format default: BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Gagal memuat citra dari {image_path}")

    # 2. Menampilkan dimensi citra (tinggi, lebar, kanal)
    tinggi, lebar, kanal = img_bgr.shape
    print(f"File citra     : {image_path}")
    print(f"Dimensi (shape): {img_bgr.shape}")
    print(f"- Tinggi       : {tinggi} piksel")
    print(f"- Lebar        : {lebar} piksel")
    print(f"- Kanal Warna  : {kanal} (BGR)")
    print("=" * 50)

    # 3. Konversi format BGR ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4. Visualisasi perbandingan BGR vs RGB dengan Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Tampilan salah (BGR ditampilkan seolah-olah RGB)
    axes[0].imshow(img_bgr)
    axes[0].set_title("Citra Mentah OpenCV (BGR)\n[Kesalahan Umum: Warna Terbalik]", fontsize=10)
    axes[0].axis("off")

    # Tampilan benar (RGB)
    axes[1].imshow(img_rgb)
    axes[1].set_title(f"Citra Dikonversi ke RGB\n[Tampilan Warna Alami] ({lebar}x{tinggi})", fontsize=10)
    axes[1].axis("off")

    plt.suptitle("Praktikum 2: Perbandingan Kanal Warna OpenCV vs Matplotlib", fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs("hasil", exist_ok=True)
    output_path = "hasil/p2_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hasil visualisasi disimpan ke: {output_path}")

if __name__ == "__main__":
    main()
