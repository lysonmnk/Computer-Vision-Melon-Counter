"""
Mini-Project: Prototipe Penghitung Buah Melon (CV Melon Counter).
- Menerima sumber masukan citra, file video, atau webcam (--source).
- Menggunakan model YOLO (models/best.pt atau model pretrained).
- Memfilter prediksi khusus kelas 'melon' dengan ambang batas confidence (default: 0.25).
- Menghitung jumlah buah melon yang terdeteksi per frame.
- Menampilkan dashboard penghitung visual dan menyimpan hasil ke folder hasil/.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="CV Melon Counter - Deteksi dan Penghitung Buah Melon")
    parser.add_argument("--source", type=str, default="data/sample.jpg", help="Path ke citra, file video, atau '0' untuk webcam")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path bobot model YOLO (default: models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang batas confidence deteksi (default: 0.25)")
    parser.add_argument("--save-dir", type=str, default="hasil", help="Folder penyimpanan hasil deteksi (default: hasil)")
    return parser.parse_args()

def resolve_model_weights(preferred_path):
    candidates = [
        preferred_path,
        "models/best.pt",
        "runs/melon/baseline/weights/best.pt",
        "yolo11n.pt",
        "yolo26n.pt",
        "yolov8n.pt"
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return "yolo11n.pt"

def draw_dashboard(frame, melon_count, confidence_threshold, fps=None):
    h, w = frame.shape[:2]
    # Buat panel dashboard semi-transparan di sudut kiri atas
    overlay = frame.copy()
    box_w = min(360, w - 20)
    box_h = 100
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Garis tepi kartu
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 200, 100), 2)

    # Teks informasi
    cv2.putText(frame, "CV MELON COUNTER", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2)
    count_str = f"JUMLAH MELON: {melon_count}"
    cv2.putText(frame, count_str, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)

    meta_str = f"Min Conf: {confidence_threshold:.2f}"
    if fps is not None:
        meta_str += f" | {fps:.1f} FPS"
    cv2.putText(frame, meta_str, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def process_image(model, image_path, conf_thresh, save_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File citra tidak ditemukan: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal memuat citra: {image_path}")

    # Jalankan inferensi
    results = model.predict(source=img, conf=conf_thresh, verbose=False)
    result = results[0]

    # Filter kelas melon (biasanya class_id 0 pada custom model, atau nama 'melon')
    melon_count = 0
    annotated = img.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, str(cls_id)).lower()
        conf = float(box.conf[0].item())

        # Model custom mendefinisikan 0: melon. Jika model pretrained COCO, deteksi objek buah/apel/oranye/bola sebagai melon
        is_melon = (cls_name == "melon" or cls_id == 0 or "melon" in cls_name)
        if not is_melon and len(model.names) > 3:
            # Fallback untuk model pretrained umum
            if cls_name in ["apple", "orange", "sports ball", "frisbee", "bowl"]:
                is_melon = True

        if is_melon and conf >= conf_thresh:
            melon_count += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # Bounding box
            color = (0, 220, 80) # Hijau cerah
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Label box
            label = f"melon #{melon_count} ({conf:.2f})"
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + t_w + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Tambahkan dashboard
    draw_dashboard(annotated, melon_count, conf_thresh)

    # Simpan output
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(save_dir, f"hitung_{base_name}.jpg")
    cv2.imwrite(out_file, annotated)

    print("=" * 50)
    print(f"Hasil Penghitungan Citra : {image_path}")
    print(f"Jumlah Melon Terhitung  : {melon_count}")
    print(f"Citra Hasil Disimpan ke  : {out_file}")
    print("=" * 50)
    return melon_count, out_file

def process_video_or_camera(model, source_str, conf_thresh, save_dir):
    is_webcam = (source_str == "0" or source_str.isdigit())
    cam_index = int(source_str) if is_webcam else None

    if is_webcam:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Webcam index {cam_index} tidak dapat dibuka. Beralih ke video cadangan data/video.mp4")
            source_str = "data/video.mp4"
            is_webcam = False
            cap = cv2.VideoCapture(source_str)
    else:
        if not os.path.exists(source_str):
            raise FileNotFoundError(f"File video tidak ditemukan: {source_str}")
        cap = cv2.VideoCapture(source_str)

    if not cap.isOpened():
        raise RuntimeError("Gagal membuka sumber video.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(save_dir, exist_ok=True)
    out_video_path = os.path.join(save_dir, "hitung_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (w, h))

    print("=" * 50)
    print(f"Memproses video: {source_str} ({w}x{h} @ {fps_in:.1f} FPS)")
    print("Tekan 'q' untuk berhenti lebih awal.")
    print("=" * 50)

    frame_idx = 0
    total_melons_detected = 0
    start_time = time.time()
    last_annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        result = results[0]

        melon_count = 0
        annotated = frame.copy()

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id)).lower()
            conf = float(box.conf[0].item())

            is_melon = (cls_name == "melon" or cls_id == 0 or "melon" in cls_name)
            if not is_melon and len(model.names) > 3:
                if cls_name in ["apple", "orange", "sports ball"]:
                    is_melon = True

            if is_melon and conf >= conf_thresh:
                melon_count += 1
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.putText(annotated, f"melon ({conf:.2f})", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

        total_melons_detected += melon_count
        elapsed = time.time() - start_time
        curr_fps = frame_idx / elapsed if elapsed > 0 else fps_in

        draw_dashboard(annotated, melon_count, conf_thresh, fps=curr_fps)
        writer.write(annotated)
        last_annotated_frame = annotated

        # Coba tampilkan GUI jika didukung
        try:
            cv2.imshow("CV Melon Counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Proses dihentikan oleh pengguna.")
                break
        except Exception:
            pass

    cap.release()
    writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Simpan juga frame terakhir sebagai citra hasil untuk inspeksi cepat
    if last_annotated_frame is not None:
        last_frame_path = os.path.join(save_dir, "hitung_video_frame.jpg")
        cv2.imwrite(last_frame_path, last_annotated_frame)

    avg_melons = total_melons_detected / frame_idx if frame_idx > 0 else 0
    print("=" * 50)
    print(f"Total Frame Diproses     : {frame_idx}")
    print(f"Rata-rata Melon / Frame  : {avg_melons:.1f}")
    print(f"Video Hasil Tersimpan di : {out_video_path}")
    print("=" * 50)

def main():
    args = parse_args()
    model_weight = resolve_model_weights(args.weights)
    print(f"Memuat model detektor: {model_weight}")
    model = YOLO(model_weight)

    # Identifikasi apakah sumber adalah citra atau video
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    _, ext = os.path.splitext(args.source)

    if ext.lower() in img_extensions:
        process_image(model, args.source, args.conf, args.save_dir)
    else:
        process_video_or_camera(model, args.source, args.conf, args.save_dir)

if __name__ == "__main__":
    main()
