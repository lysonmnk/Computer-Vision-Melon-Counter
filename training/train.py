"""
Training Model YOLO Custom untuk Deteksi Buah Melon.
- Melatih model YOLO menggunakan dataset/data.yaml.
- Menyimpan hasil training ke runs/melon/baseline.
- Menyalin bobot terbaik ke models/best.pt.
- Menyalin grafik metrik dan matriks konfusi ke hasil/results.png dan hasil/confusion_matrix.png.
"""

import os
import shutil
import argparse
from ultralytics import YOLO

def find_base_model():
    for name in ["yolo11n.pt", "yolo26n.pt", "yolov8n.pt"]:
        if os.path.exists(name):
            return name
    return "yolo11n.pt"

def main():
    parser = argparse.ArgumentParser(description="Pelatihan Model YOLO Melon Counter")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch pelatihan (default: 10)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi citra input (default: 640)")
    parser.add_argument("--weights", type=str, default=None, help="Bobot dasar YOLO (default: auto)")
    parser.add_argument("--project", type=str, default="runs/melon", help="Folder proyek training")
    parser.add_argument("--name", type=str, default="baseline", help="Nama eksperimen")
    args = parser.parse_args()

    print("=" * 60)
    print("MEMULAI TRAINING MODEL YOLO CUSTOM: MELON DETECTOR")
    print("=" * 60)

    base_model = args.weights if args.weights else find_base_model()
    print(f"Base model    : {base_model}")
    print(f"Dataset config: dataset/data.yaml")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch}")
    print(f"Image Size    : {args.imgsz}")
    print(f"Project Run   : {args.project}/{args.name}")
    print("-" * 60)

    model = YOLO(base_model)
    model.train(
        data="dataset/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        workers=0 # Menghindari issue multiprocessing di Windows
    )

    print("=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)

    # 1. Simpan bobot ke models/best.pt
    os.makedirs("models", exist_ok=True)
    os.makedirs("hasil", exist_ok=True)
    
    run_dir = str(model.trainer.save_dir) if hasattr(model, 'trainer') and model.trainer else os.path.join(args.project, args.name)
    best_weight_path = os.path.join(run_dir, "weights", "best.pt")
    target_weight_path = "models/best.pt"

    if os.path.exists(best_weight_path):
        shutil.copy2(best_weight_path, target_weight_path)
        print(f"Bobot terbaik disalin ke: {target_weight_path}")
    else:
        last_weight = os.path.join(run_dir, "weights", "last.pt")
        if os.path.exists(last_weight):
            shutil.copy2(last_weight, target_weight_path)
            print(f"Bobot terakhir disalin ke: {target_weight_path}")

    # 2. Salin grafik hasil evaluasi ke folder hasil/
    results_png = os.path.join(run_dir, "results.png")
    if os.path.exists(results_png):
        shutil.copy2(results_png, "hasil/results.png")
        print("Grafik training disalin ke: hasil/results.png")

    cm_png = os.path.join(run_dir, "confusion_matrix.png")
    if os.path.exists(cm_png):
        shutil.copy2(cm_png, "hasil/confusion_matrix.png")
        print("Confusion matrix disalin ke: hasil/confusion_matrix.png")
    else:
        cm_norm = os.path.join(run_dir, "confusion_matrix_normalized.png")
        if os.path.exists(cm_norm):
            shutil.copy2(cm_norm, "hasil/confusion_matrix.png")
            print("Confusion matrix disalin ke: hasil/confusion_matrix.png")

    print("=" * 60)

if __name__ == "__main__":
    main()
