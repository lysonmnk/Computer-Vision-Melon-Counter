"""
Validasi dan Evaluasi Model YOLO Melon Counter.
- Mengevaluasi model models/best.pt pada validation split dataset/data.yaml.
- Menghitung dan menampilkan Precision, Recall, mAP@50, mAP@50-95, dan latency.
- Menampilkan rincian metrik per kelas (melon, daun, buah_rusak).
"""

import os
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
    print("VALIDASI DAN EVALUASI MODEL YOLO CUSTOM")
    print("=" * 60)

    model_path = find_model()
    print(f"Menggunakan bobot model: {model_path}")
    print(f"Dataset validasi       : dataset/data.yaml (split: val)")
    print("-" * 60)

    model = YOLO(model_path)
    metrics = model.val(
        data="dataset/data.yaml",
        split="val",
        imgsz=640,
        batch=4,
        workers=0
    )

    print("=" * 60)
    print("HASIL EVALUASI MODEL KESELURUHAN")
    print("=" * 60)

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    # Speed metrics
    preprocess_ms = metrics.speed.get("preprocess", 0.0)
    inference_ms = metrics.speed.get("inference", 0.0)
    postprocess_ms = metrics.speed.get("loss", 0.0) + metrics.speed.get("postprocess", 0.0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print(f"Precision (P)       : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall (R)          : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"mAP@50              : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP@50-95           : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"Waktu Inferensi     : {inference_ms:.2f} ms/citra (Total: {total_ms:.2f} ms)")
    print(f"Estimasi Throughput : {fps:.1f} FPS (pada CPU)")
    print("-" * 60)

    print("RINCIAN EVALUASI PER KELAS:")
    print(f"{'Kelas':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10}")
    print("-" * 55)

    class_names = ["melon", "daun", "buah_rusak"]
    class_p = metrics.box.p
    class_r = metrics.box.r
    class_ap50 = metrics.box.ap50

    for i, name in enumerate(class_names):
        p_val = class_p[i] if i < len(class_p) else 0.0
        r_val = class_r[i] if i < len(class_r) else 0.0
        ap_val = class_ap50[i] if i < len(class_ap50) else 0.0
        print(f"{name:<15} | {p_val:<10.4f} | {r_val:<10.4f} | {ap_val:<10.4f}")

    print("=" * 60)

if __name__ == "__main__":
    main()
