"""
Skrip pembuat dataset dan data sampel buah melon sintetis realistis
untuk pelatihan YOLO dan pengujian counter.
"""

import os
import random
import cv2
import numpy as np

# Seed untuk konsistensi
random.seed(42)
np.random.seed(42)

def draw_leaf(img, center, size, angle):
    # Menggambar daun melon hijau berbentuk poligonal
    x, y = center
    pts = []
    num_pts = 7
    for i in range(num_pts):
        theta = angle + i * (2 * np.pi / num_pts)
        r = size * (0.7 + 0.3 * np.sin(3 * theta))
        px = int(x + r * np.cos(theta))
        py = int(y + r * np.sin(theta))
        pts.append([px, py])
    pts = np.array(pts, np.int32)
    color = (random.randint(20, 60), random.randint(110, 180), random.randint(30, 80)) # BGR
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], isClosed=True, color=(15, 90, 20), thickness=2)
    # Tulang daun
    cv2.line(img, (x, y), (int(x + size * np.cos(angle)), int(y + size * np.sin(angle))), (10, 80, 15), 1)

def draw_melon(img, center, rx, ry, class_id=0):
    # Menggambar buah melon dengan tekstur jaring dan shading
    h, w = img.shape[:2]
    x, y = center
    
    # Base color
    if class_id == 0:
        # Melon sehat (krem kehijauan / cantaloupe)
        base_color = (random.randint(90, 140), random.randint(170, 210), random.randint(180, 225))
    else:
        # Buah rusak (kecoklatan, bercak hitam/busuk)
        base_color = (random.randint(50, 90), random.randint(80, 120), random.randint(100, 140))
        
    # Draw melon body
    cv2.ellipse(img, (x, y), (rx, ry), 0, 0, 360, base_color, -1)
    
    # Gradient shading / 3D depth
    overlay = img.copy()
    cv2.ellipse(overlay, (x - rx // 4, y - ry // 4), (rx // 2, ry // 2), 0, 0, 360, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    
    # Shadow at bottom
    overlay = img.copy()
    cv2.ellipse(overlay, (x + rx // 5, y + ry // 4), (rx // 2, ry // 2), 0, 0, 360, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    
    # Texture netting (jala melon)
    for _ in range(12):
        pt1 = (x + random.randint(-rx + 5, rx - 5), y + random.randint(-ry + 5, ry - 5))
        pt2 = (pt1[0] + random.randint(-15, 15), pt1[1] + random.randint(-15, 15))
        net_color = (200, 230, 230) if class_id == 0 else (60, 60, 80)
        cv2.line(img, pt1, pt2, net_color, 1)

    if class_id == 2:
        # Blemish / rot spot untuk buah rusak
        bx = x + random.randint(-rx // 2, rx // 2)
        by = y + random.randint(-ry // 2, ry // 2)
        cv2.circle(img, (bx, by), random.randint(8, 16), (20, 30, 45), -1)
        
    # Tangkai melon
    stem_len = int(ry * 0.4)
    cv2.line(img, (x, y - ry), (x + random.randint(-5, 5), y - ry - stem_len), (30, 100, 50), 3)

def generate_image(width=640, height=640, num_melons=3, num_leaves=4, damaged_prob=0.2):
    # Background tanah / kebun
    bg_color = (random.randint(35, 60), random.randint(50, 85), random.randint(65, 105)) # soil tone
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Tambah noise tanah
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Sulur / tali rambatan
    for _ in range(5):
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        cv2.line(img, pt1, pt2, (20, 80, 30), 2)
        
    boxes = [] # (class_id, x_center, y_center, w, h)
    
    # Daun latar belakang
    for _ in range(num_leaves):
        lx = random.randint(50, width - 50)
        ly = random.randint(50, height - 50)
        lsize = random.randint(40, 80)
        draw_leaf(img, (lx, ly), lsize, random.uniform(0, 2 * np.pi))
        # Sebagian daun dianotasi sebagai kelas 1 (daun)
        if random.random() < 0.5:
            bw = min(lsize * 2, width)
            bh = min(lsize * 2, height)
            boxes.append((1, lx / width, ly / height, bw / width, bh / height))

    # Buah melon
    placed_melons = []
    for _ in range(num_melons):
        # Cari posisi agar tidak terlalu tumpang tindih total
        for _ in range(20):
            mx = random.randint(90, width - 90)
            my = random.randint(90, height - 90)
            rx = random.randint(45, 75)
            ry = int(rx * random.uniform(1.0, 1.25))
            
            # Cek overlap dengan melon lain
            overlap = False
            for (ox, oy, orx, ory) in placed_melons:
                dist = np.hypot(mx - ox, my - oy)
                if dist < (rx + orx) * 0.7:
                    overlap = True
                    break
            if not overlap or len(placed_melons) == 0:
                break
                
        placed_melons.append((mx, my, rx, ry))
        is_damaged = (random.random() < damaged_prob)
        cls_id = 2 if is_damaged else 0
        
        draw_melon(img, (mx, my), rx, ry, class_id=cls_id)
        
        # Anotasi bbox
        bw = (rx * 2 + 6) / width
        bh = (ry * 2 + int(ry * 0.4) + 6) / height
        bx = mx / width
        by = (my - int(ry * 0.2)) / height
        # Clip
        bx = max(0.01, min(0.99, bx))
        by = max(0.01, min(0.99, by))
        bw = max(0.02, min(0.98, bw))
        bh = max(0.02, min(0.98, bh))
        boxes.append((cls_id, bx, by, bw, bh))

    # Beberapa daun menutupi sebagian buah (occlusion)
    for _ in range(random.randint(1, 3)):
        lx = random.randint(50, width - 50)
        ly = random.randint(50, height - 50)
        lsize = random.randint(35, 65)
        draw_leaf(img, (lx, ly), lsize, random.uniform(0, 2 * np.pi))
        
    return img, boxes

def save_sample_and_dataset():
    # 1. Simpan data/sample.jpg (4 melon jelas)
    sample_img, _ = generate_image(640, 640, num_melons=4, num_leaves=5, damaged_prob=0.0)
    cv2.imwrite("data/sample.jpg", sample_img)
    print("Berhasil membuat data/sample.jpg")

    # 2. Buat train dataset (24 gambar)
    for i in range(1, 25):
        img, bboxes = generate_image(640, 640, num_melons=random.randint(2, 6), num_leaves=random.randint(3, 7))
        img_path = f"dataset/images/train/melon_train_{i:03d}.jpg"
        lbl_path = f"dataset/labels/train/melon_train_{i:03d}.txt"
        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            for cls_id, bx, by, bw, bh in bboxes:
                f.write(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
    print("Berhasil membuat 24 gambar training dataset")

    # 3. Buat val dataset (6 gambar)
    for i in range(1, 7):
        img, bboxes = generate_image(640, 640, num_melons=random.randint(2, 5), num_leaves=random.randint(3, 6))
        img_path = f"dataset/images/val/melon_val_{i:03d}.jpg"
        lbl_path = f"dataset/labels/val/melon_val_{i:03d}.txt"
        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            for cls_id, bx, by, bw, bh in bboxes:
                f.write(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
    print("Berhasil membuat 6 gambar validation dataset")

    # 4. Buat video simulasi kebun melon (data/video.mp4, 60 frame = 2 detik @ 30 fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/video.mp4', fourcc, 30.0, (640, 480))
    # Buat panorama kebun lalu pan kamera
    big_img, _ = generate_image(1280, 480, num_melons=10, num_leaves=15, damaged_prob=0.1)
    for f in range(60):
        # Geser jendela kamera dari kiri ke kanan
        shift_x = int(f * (640 / 60))
        frame = big_img[:, shift_x:shift_x + 640]
        out.write(frame)
    out.release()
    print("Berhasil membuat data/video.mp4")

if __name__ == "__main__":
    save_sample_and_dataset()
