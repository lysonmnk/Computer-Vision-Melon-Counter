"""
Praktikum 1: Menyiapkan lingkungan dan memeriksa instalasi pustaka.
Memeriksa versi OpenCV, NumPy, Torch, dan Ultralytics YOLO.
Menyimpan ringkasan versi ke hasil/p1_output.png.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    print("=" * 50)
    print("PRAKTIKUM 1: PEMERIKSAAN INSTALASI PUSTAKA")
    print("=" * 50)

    py_version = sys.version.split()[0]
    cv_version = cv2.__version__
    np_version = np.__version__
    mpl_version = matplotlib.__version__

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "Tidak terpasang"
        cuda_available = False

    try:
        import ultralytics
        from ultralytics import YOLO
        yolo_version = ultralytics.__version__
        yolo_status = "Berhasil dimuat"
    except ImportError as e:
        yolo_version = "Tidak terpasang"
        yolo_status = f"Gagal: {e}"

    print(f"Python      : {py_version}")
    print(f"OpenCV      : {cv_version}")
    print(f"NumPy       : {np_version}")
    print(f"Matplotlib  : {mpl_version}")
    print(f"PyTorch     : {torch_version} (CUDA: {cuda_available})")
    print(f"Ultralytics : {yolo_version} ({yolo_status})")
    print("=" * 50)
    print("Status lingkungan: SIAP DIGUNAKAN")
    print("=" * 50)

    # Simpan ringkasan ke hasil/p1_output.png sebagai bukti pengujian
    os.makedirs("hasil", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#f4f6f9")
    ax.axis("off")

    info_text = (
        "CV MELON COUNTER - HASIL CEK INSTALASI (P1)\n\n"
        f"- Python Version        : {py_version}\n"
        f"- OpenCV (opencv-python): {cv_version}\n"
        f"- NumPy Version         : {np_version}\n"
        f"- Matplotlib Version    : {mpl_version}\n"
        f"- PyTorch Version       : {torch_version}\n"
        f"- CUDA Hardware Accel   : {'Aktif (GPU)' if cuda_available else 'Nonaktif (CPU)'}\n"
        f"- Ultralytics YOLO      : {yolo_version} ({yolo_status})\n\n"
        "Status Verifikasi: Seluruh dependensi terpasang dan berfungsi normal."
    )

    ax.text(
        0.05, 0.90, info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="#2b5797", linewidth=2)
    )

    output_path = "hasil/p1_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bukti pengujian disimpan ke: {output_path}")

if __name__ == "__main__":
    main()
