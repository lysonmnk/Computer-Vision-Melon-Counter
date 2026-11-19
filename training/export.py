"""
Ekspor Model YOLO ke Format ONNX.
- Mengonversi bobot PyTorch (.pt) ke Open Neural Network Exchange (.onnx).
- Siap digunakan untuk deployment efisien pada perangkat edge atau runtime ONNX.
"""

import os
import shutil
from ultralytics import YOLO

def find_model():
    if os.path.exists("models/best.pt"):
        return "models/best.pt"
    if os.path.exists("runs/melon/baseline/weights/best.pt"):
        return "runs/melon/baseline/weights/best.pt"
    if os.path.exists("yolo11n.pt"):
        return "yolo11n.pt"
    return "yolo11n.pt"

def main():
    print("=" * 60)
    print("EKSPOR MODEL YOLO KE FORMAT ONNX")
    print("=" * 60)

    model_path = find_model()
    print(f"Memuat model untuk diekspor: {model_path}")
    model = YOLO(model_path)

    # Ekspor ke ONNX
    print("Memulai proses konversi ke format ONNX...")
    exported_file = model.export(format="onnx", imgsz=640, dynamic=False)
    print(f"Model berhasil diekspor ke: {exported_file}")

    # Salin ke folder models/ jika belum di sana
    os.makedirs("models", exist_ok=True)
    target_onnx = "models/best.onnx"
    if os.path.exists(exported_file) and os.path.abspath(exported_file) != os.path.abspath(target_onnx):
        shutil.copy2(exported_file, target_onnx)
        print(f"Salinan ONNX disimpan di: {target_onnx}")

    print("=" * 60)

if __name__ == "__main__":
    main()
