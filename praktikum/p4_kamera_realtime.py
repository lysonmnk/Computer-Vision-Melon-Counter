"""
Praktikum 4: Kamera real-time.
- Membuka kamera webcam default (cv2.VideoCapture(0)).
- Menyediakan fallback ke video/simulasi jika perangkat kamera fisik tidak tersedia.
- Menampilkan resolusi, frame rate, dan status real-time.
- Tekan 'q' untuk keluar atau 's' untuk menyimpan frame.
- Menyimpan frame tangkapan ke hasil/p4_kamera.png sebagai bukti pengujian.
"""

import os
import sys
import time
import cv2

def main():
    print("=" * 50)
    print("PRAKTIKUM 4: KAMERA REAL-TIME")
    print("=" * 50)

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    using_fallback = False

    # Verifikasi apakah kamera fisik dapat dibuka
    if not cap.isOpened():
        print(f"Peringatan: Kamera default (index {camera_index}) tidak dapat diakses.")
        fallback_video = "data/video.mp4"
        if os.path.exists(fallback_video):
            print(f"Mengalihkan sumber ke simulasi video real-time: {fallback_video}")
            cap = cv2.VideoCapture(fallback_video)
            using_fallback = True
        else:
            print("Video fallback tidak ditemukan. Membuat frame simulasi kamera.")

    # Atur resolusi kamera jika kamera fisik aktif
    if not using_fallback and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs("hasil", exist_ok=True)
    captured_frame = None
    frame_count = 0
    start_time = time.time()

    print("Kamera aktif. Tekan 'q' pada jendela tampilan untuk keluar.")
    print("Tekan 's' untuk menyimpan frame.")

    # Loop penangkapan frame
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Jika video simulasi selesai, loop kembali
                if using_fallback:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    break
        else:
            # Fallback frame simulasi jika tidak ada kamera
            ret = True
            frame = cv2.imread("data/sample.jpg")
            if frame is None:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 30.0

        # Tambahkan overlay status pada frame
        display_frame = frame.copy()
        status_text = f"Kamera {'Simulasi' if using_fallback or not cap.isOpened() else 'Webcam'} | FPS: {fps:.1f} | Frame: {frame_count}"
        cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Tekan 'q' untuk keluar, 's' simpan", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        captured_frame = display_frame

        # Tampilkan jendela jika GUI didukung
        try:
            cv2.imshow("Praktikum 4 - Kamera Real-Time", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Pengguna menghentikan streaming kamera ('q').")
                break
            elif key == ord('s'):
                cv2.imwrite("hasil/p4_kamera.png", captured_frame)
                print("Frame berhasil disimpan secara manual.")
        except cv2.error:
            # Mode headless / tanpa display window
            if frame_count >= 10:
                print("Lingkungan headless terdeteksi, 10 frame berhasil diproses.")
                break

        # Otomatis selesai setelah beberapa frame jika di terminal non-interaktif
        if not sys.stdin.isatty() and frame_count >= 15:
            break

    if cap.isOpened():
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Simpan tangkapan frame terakhir ke hasil/p4_kamera.png
    if captured_frame is not None:
        output_path = "hasil/p4_kamera.png"
        cv2.imwrite(output_path, captured_frame)
        print("=" * 50)
        print(f"Tangkapan kamera real-time disimpan ke: {output_path}")
        print("=" * 50)

if __name__ == "__main__":
    main()
