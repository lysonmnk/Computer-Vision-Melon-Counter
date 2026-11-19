"""
Praktikum 3: Preprocessing citra.
- Resize 640x640 (cv2.resize) -> Menyamakan ukuran input
- Grayscale (cv2.cvtColor) -> Menyederhanakan kanal warna
- Gaussian Blur 5x5 (cv2.GaussianBlur) -> Mengurangi noise
- Canny Edge Detection 80, 160 (cv2.Canny) -> Mendeteksi kontur/tepi
Output: hasil/resized.jpg dan hasil/edges.jpg
"""

import os
import cv2

def main():
    input_path = "data/sample.jpg"
    print("=" * 50)
    print("PRAKTIKUM 3: PREPROCESSING CITRA")
    print("=" * 50)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File sampel tidak ditemukan: {input_path}")

    # Baca citra asli
    img = cv2.imread(input_path)
    print(f"Ukuran asli citra: {img.shape}")

    # 1. Resize ke 640x640
    target_size = (640, 640)
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    print(f"1. Resize selesai: {resized.shape}")

    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    print(f"2. Grayscale selesai: {gray.shape}")

    # 3. Gaussian Blur (kernel 5x5, sigmaX 0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("3. Gaussian Blur (5x5) selesai: noise berhasil diredam")

    # 4. Canny Edge Detection (threshold 80, 160)
    edges = cv2.Canny(blurred, 80, 160)
    print("4. Canny Edge Detection (80, 160) selesai")

    # Simpan output
    os.makedirs("hasil", exist_ok=True)
    out_resized = "hasil/resized.jpg"
    out_edges = "hasil/edges.jpg"

    cv2.imwrite(out_resized, resized)
    cv2.imwrite(out_edges, edges)

    print("=" * 50)
    print(f"Output tersimpan:")
    print(f"- {out_resized}")
    print(f"- {out_edges}")
    print("=" * 50)

if __name__ == "__main__":
    main()
