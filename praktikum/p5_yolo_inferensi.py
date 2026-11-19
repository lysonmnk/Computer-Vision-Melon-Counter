"""
Praktikum 5: Inferensi objek dengan YOLO.
- Memuat model YOLO pretrained (yolo11n.pt / yolo26n.pt).
- Menjalankan inferensi pada data/sample.jpg dengan ambang batas conf=0.25.
- Mengekstrak dan mencetak nama kelas, confidence score, dan koordinat bounding box.
- Menyimpan citra hasil anotasi deteksi ke hasil/p5_deteksi.jpg.
"""

import os
import cv2
from ultralytics import YOLO

def find_pretrained_model():
    candidates = ["yolo11n.pt", "yolo26n.pt", "yolov8n.pt"]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return "yolo11n.pt"

def main():
    print("=" * 50)
    print("PRAKTIKUM 5: INFERENSI OBJEK DENGAN YOLO")
    print("=" * 50)

    image_path = "data/sample.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File citra sampel tidak ditemukan: {image_path}")

    model_weight = find_pretrained_model()
    print(f"Memuat model YOLO pretrained: {model_weight}")
    model = YOLO(model_weight)

    # Jalankan inferensi
    conf_threshold = 0.25
    print(f"Menjalankan inferensi pada {image_path} dengan conf={conf_threshold}...")
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # Analisis hasil deteksi
    result = results[0]
    boxes = result.boxes

    print("-" * 50)
    print(f"Total objek terdeteksi: {len(boxes)}")
    print("-" * 50)

    for idx, box in enumerate(boxes, start=1):
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(val, 2) for val in box.xyxy[0].tolist()]

        print(f"[{idx}] Objek: {cls_name}")
        print(f"    Confidence : {conf:.4f} ({conf * 100:.1f}%)")
        print(f"    BoundingBox: [{x1}, {y1}, {x2}, {y2}]")

    # Render hasil anotasi
    annotated_img = result.plot()

    os.makedirs("hasil", exist_ok=True)
    output_path = "hasil/p5_deteksi.jpg"
    cv2.imwrite(output_path, annotated_img)

    print("=" * 50)
    print(f"Hasil citra teranotasi disimpan ke: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
