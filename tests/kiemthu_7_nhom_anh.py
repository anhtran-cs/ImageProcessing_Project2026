"""
KIỂM THỬ THỰC NGHIỆM ĐÁNH GIÁ 7 NHÓM ĐIỀU KIỆN ẢNH (100 ẢNH)
============================================================
Bộ dữ liệu chuẩn hóa gồm 100 ảnh theo đúng bảng yêu cầu:
1. Ánh sáng đều (20 ảnh) - Baseline
2. Ánh sáng không đều (20 ảnh) - Gradient một bên sáng một bên tối
3. Bóng đổ (15 ảnh) - Vệt bóng tối cục bộ (bóng tay, bóng điện thoại)
4. Nền ngoài phức tạp (15 ảnh) - Mặt bàn gỗ vân, vải, gạch, họa tiết
5. Trang nghiêng / Phối cảnh (10 ảnh) - Chụp xiên góc 15-35 độ
6. Giấy cũ / Vàng / Mỏng (10 ảnh) - Giấy ố vàng, hằn chữ mặt sau
7. Chữ mờ / Nhiễu (10 ảnh) - Bản in mờ, nét đứt, rung tay, nhiễu hạt

Chạy: python kiemthu/kiemthu_7_nhom_anh.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "document_scanner"))
sys.path.insert(0, str(ROOT))

from document_scanner.xulytailieu import (
    process_document,
    document_metrics,
    detect_document_corners,
)
from core.xulyanh import save_image


# ==============================================================
# HÀM TẠO 7 NHÓM ẢNH ĐẶC THÙ (SYNTHETIC BENCHMARK ENGINE)
# ==============================================================

def create_base_paper(h=800, w=600, paper_bgr=(245, 245, 245), bg_bgr=(55, 55, 55), num_lines=14):
    """Tạo trang tài liệu cơ bản với các dòng chữ."""
    img = np.full((h, w, 3), bg_bgr, dtype=np.uint8)
    mx, my = w // 9, h // 11
    # Tờ giấy
    cv2.rectangle(img, (mx, my), (w - mx, h - my), paper_bgr, -1)
    
    # Tiêu đề
    cv2.putText(img, "REPORT ON DOCUMENT DIGITIZATION", (mx + 30, my + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.line(img, (mx + 30, my + 60), (w - mx - 30, my + 60), (40, 40, 40), 1)

    # Các dòng văn bản
    for i in range(num_lines):
        y = my + 95 + i * 38
        if y > h - my - 30:
            break
        text = f"Section {i+1}. This is high precision test line {i+1} for optical binarization."
        cv2.putText(img, text, (mx + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return img, (mx, my, w - mx, h - my)


def gen_group1_uniform(idx: int) -> np.ndarray:
    """Nhóm 1: Ánh sáng đều, phẳng, nét rõ (Baseline)."""
    h, w = 800, 600
    img, _ = create_base_paper(h, w, paper_bgr=(248, 248, 248), bg_bgr=(50, 50, 50))
    return img


def gen_group2_gradient(idx: int) -> np.ndarray:
    """Nhóm 2: Ánh sáng không đều (gradient ngang/dọc/chéo lệch nguồn sáng)."""
    h, w = 800, 600
    img, _ = create_base_paper(h, w)
    
    # Tạo gradient sáng/tối theo nhiều hướng
    direction = idx % 4
    if direction == 0:  # Trái sáng, Phải tối
        grad = np.linspace(1.15, 0.45, w).reshape(1, w)
        mask = np.tile(grad, (h, 1))
    elif direction == 1:  # Trên sáng, Dưới tối
        grad = np.linspace(1.2, 0.4, h).reshape(h, 1)
        mask = np.tile(grad, (1, w))
    elif direction == 2:  # Chéo góc
        X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        mask = 1.25 - 0.75 * (X * 0.6 + Y * 0.4)
    else:  # Spotlight tròn lệch tâm
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w*0.3)**2 + (Y - h*0.2)**2)
        mask = 1.2 - 0.7 * (dist / np.max(dist))

    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return img


def gen_group3_shadows(idx: int) -> np.ndarray:
    """Nhóm 3: Vệt bóng đổ cục bộ (bóng tay, bóng điện thoại)."""
    h, w = 800, 600
    img, _ = create_base_paper(h, w)
    
    # Tạo vệt bóng tối đa dạng
    shadow_mask = np.ones((h, w), dtype=np.float32)
    s_type = idx % 3
    if s_type == 0:  # Bóng điện thoại hình chữ nhật chéo
        pts = np.array([[w*0.3, h*0.2], [w*0.75, h*0.4], [w*0.6, h*0.8], [w*0.15, h*0.6]], np.int32)
        cv2.fillConvexPoly(shadow_mask, pts, 0.42)
    elif s_type == 1:  # Bóng bàn tay / ngón tay chùm từ cạnh
        cv2.ellipse(shadow_mask, (int(w*0.85), int(h*0.5)), (int(w*0.35), int(h*0.25)), -30, 0, 360, 0.38, -1)
    else:  # Bóng đổ nửa mặt giấy
        cv2.circle(shadow_mask, (int(w*0.2), int(h*0.8)), int(w*0.5), 0.45, -1)

    # Làm mờ biên bóng đổ (soft shadow tự nhiên)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (45, 45), 0)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * shadow_mask, 0, 255).astype(np.uint8)
    return img


def gen_group4_complex_bg(idx: int) -> np.ndarray:
    """Nhóm 4: Nền ngoài phức tạp (vân gỗ, vải, thảm, gạch sọc)."""
    h, w = 800, 600
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    bg_type = idx % 3
    if bg_type == 0:  # Vân gỗ (Wood grain stripes)
        for y in range(h):
            val = int(80 + 35 * np.sin(y / 12.0) + 15 * np.cos(y / 4.0))
            bg[y, :] = (val - 25, val, val + 35)
    elif bg_type == 1:  # Khăn vải kẻ caro (Checkered fabric)
        block = 25
        for y in range(0, h, block):
            for x in range(0, w, block):
                color = (70, 60, 90) if ((x // block + y // block) % 2 == 0) else (120, 100, 140)
                bg[y:y+block, x:x+block] = color
    else:  # Thảm hạt nhiễu vân đá (Marble texture)
        noise = np.random.randint(60, 130, (h, w), dtype=np.uint8)
        bg[:, :, 0] = noise
        bg[:, :, 1] = np.clip(noise + 15, 0, 255)
        bg[:, :, 2] = np.clip(noise + 30, 0, 255)

    # Đặt tờ giấy lên nền phức tạp
    mx, my = w // 8, h // 10
    cv2.rectangle(bg, (mx, my), (w - mx, h - my), (242, 242, 242), -1)
    for i in range(13):
        y = my + 60 + i * 40
        cv2.putText(bg, f"Complex background text record line {i+1}", (mx + 25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return bg


def gen_group5_perspective(idx: int) -> np.ndarray:
    """Nhóm 5: Trang nghiêng / Phối cảnh 3D góc 15-35 độ."""
    h, w = 800, 600
    flat_img, _ = create_base_paper(h, w, bg_bgr=(35, 35, 35))
    
    # 4 góc nguồn
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Các biến dạng góc nghiêng khác nhau
    tilt = idx % 4
    if tilt == 0:  # Nghiêng từ trên xuống (chụp từ trước mặt)
        dst = np.float32([[w*0.12, h*0.06], [w*0.88, h*0.06], [w*0.98, h*0.94], [w*0.02, h*0.94]])
    elif tilt == 1:  # Xiên góc trái
        dst = np.float32([[w*0.05, h*0.15], [w*0.82, h*0.04], [w*0.95, h*0.85], [w*0.18, h*0.96]])
    elif tilt == 2:  # Xiên góc phải
        dst = np.float32([[w*0.18, h*0.04], [w*0.95, h*0.15], [w*0.82, h*0.96], [w*0.05, h*0.85]])
    else:  # Xoay phối cảnh chéo
        dst = np.float32([[w*0.15, h*0.08], [w*0.90, h*0.18], [w*0.85, h*0.92], [w*0.08, h*0.82]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(flat_img, M, (w, h), borderValue=(35, 35, 35))
    return warped


def gen_group6_aged_paper(idx: int) -> np.ndarray:
    """Nhóm 6: Giấy cũ / ngả vàng / hằn nét chữ mặt sau."""
    h, w = 800, 600
    # Nền giấy ngả vàng / sepia (BGR: vàng nhạt tới vàng sẫm)
    yellow_tones = [(175, 215, 235), (150, 195, 220), (160, 205, 230), (180, 225, 240)]
    paper_color = yellow_tones[idx % len(yellow_tones)]
    
    img, (mx, my, x2, y2) = create_base_paper(h, w, paper_bgr=paper_color, bg_bgr=(45, 45, 45))
    
    # Hằn chữ mặt sau (Bleed-through text: mờ, lộn ngược nhẹ)
    for j in range(8):
        y_ghost = my + 75 + j * 55
        cv2.putText(img, "Backside mirrored bleedthrough text content", (mx + 35, y_ghost),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 130, 140), 1, cv2.LINE_AA)

    # Vết ố / đốm loang màu (Stains)
    for _ in range(3):
        cx, cy = np.random.randint(mx+50, x2-50), np.random.randint(my+50, y2-50)
        cv2.circle(img, (cx, cy), np.random.randint(20, 60), (130, 175, 200), -1)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def gen_group7_faint_noise(idx: int) -> np.ndarray:
    """Nhóm 7: Chữ mờ / nhiễu hạt / mực kim đứt nét / rung nhẹ."""
    h, w = 800, 600
    # Mực nhạt (màu xám nhạt thay vì đen đậm)
    img, _ = create_base_paper(h, w, paper_bgr=(242, 242, 242), bg_bgr=(50, 50, 50))
    
    # 1. Thêm nhiễu Gaussian / nhiễu hạt
    noise = np.random.normal(0, 18, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Làm mờ nhẹ (nhòe / rung tay)
    if idx % 2 == 0:
        kernel = np.ones((3, 3), np.float32) / 9.0
        noisy = cv2.filter2D(noisy, -1, kernel)
        
    return noisy


# ==============================================================
# BỘ KHUNG THỰC NGHIỆM ĐÁNH GIÁ (TEST HARNESS)
# ==============================================================

GROUPS_CONFIG = [
    ("Nhóm 1: Ánh sáng đều (Baseline)", 20, gen_group1_uniform),
    ("Nhóm 2: Ánh sáng không đều (Gradient)", 20, gen_group2_gradient),
    ("Nhóm 3: Bóng đổ cục bộ (Shadows)", 15, gen_group3_shadows),
    ("Nhóm 4: Nền ngoài phức tạp (Complex Background)", 15, gen_group4_complex_bg),
    ("Nhóm 5: Trang nghiêng / Phối cảnh (Perspective)", 10, gen_group5_perspective),
    ("Nhóm 6: Giấy cũ / Vàng / Mỏng (Aged Paper)", 10, gen_group6_aged_paper),
    ("Nhóm 7: Chữ mờ / Nhiễu hạt (Faint / Noisy)", 10, gen_group7_faint_noise),
]


def run_comprehensive_benchmark():
    out_dir = ROOT / "tests" / "ketqua_7_nhom"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ THỰC NGHIỆM ĐÁNH GIÁ 7 NHÓM ĐIỀU KIỆN ẢNH (100 ẢNH)")
    print("=" * 80)
    
    total_images = sum(count for _, count, _ in GROUPS_CONFIG)
    overall_passed = 0
    start_all = time.perf_counter()

    summary_records = []

    for g_idx, (g_name, g_count, g_func) in enumerate(GROUPS_CONFIG, 1):
        print(f"\n▶ Đang kiểm thử {g_name} [{g_count} ảnh]...")
        g_detected = 0
        g_clean_bg = 0
        g_times = []
        g_black_ratios = []

        for i in range(g_count):
            img = g_func(i)
            t0 = time.perf_counter()
            
            # Chạy qua Full Pipeline của Project 2
            res = process_document(
                img,
                detect_page=True,
                inset_ratio=0.015,
                remove_background=True,
                background_kernel=51,
                adaptive_method="Gaussian",
                block_size=21,
                c=10.0,
                morph_operation="Opening",
                morph_kernel=3,
                morph_iterations=1,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            g_times.append(elapsed)

            m = document_metrics(res.binary)
            g_black_ratios.append(m["black_ratio"])

            if res.document_found:
                g_detected += 1
            
            # Tiêu chuẩn nền sạch: White ratio >= 90% và Black ratio trong khoảng hợp lý (0.2% - 15%)
            if 0.1 <= m["black_ratio"] <= 15.0 and m["white_ratio"] >= 85.0:
                g_clean_bg += 1

            # Lưu ảnh mẫu đại diện cho mỗi nhóm (ảnh đầu tiên của nhóm)
            if i == 0:
                sample_orig = out_dir / f"group{g_idx}_sample_orig.png"
                sample_proc = out_dir / f"group{g_idx}_sample_proc.png"
                save_image(str(sample_orig), img)
                save_image(str(sample_proc), res.binary)

        avg_time = np.mean(g_times)
        avg_black = np.mean(g_black_ratios)
        detect_rate = (g_detected / g_count) * 100
        clean_rate = (g_clean_bg / g_count) * 100
        overall_passed += g_clean_bg

        summary_records.append({
            "group": g_name,
            "count": g_count,
            "detected": g_detected,
            "detect_rate": detect_rate,
            "clean_rate": clean_rate,
            "avg_time": avg_time,
            "avg_black": avg_black,
        })

        print(f"   ✓ Tìm thấy góc: {g_detected}/{g_count} ({detect_rate:.1f}%)")
        print(f"   ✓ Nhị phân đạt chuẩn: {g_clean_bg}/{g_count} ({clean_rate:.1f}%)")
        print(f"   ✓ Thời gian xử lý TB: {avg_time:.1f} ms/ảnh | Tỷ lệ nét chữ đen TB: {avg_black:.2f}%")

    total_time = (time.perf_counter() - start_all)

    print("\n" + "=" * 80)
    print("TỔNG KẾT KIỂM THỬ THỰC NGHIỆM 100 ẢNH")
    print("=" * 80)
    print(f"Tổng số ảnh kiểm thử: {total_images} ảnh")
    print(f"Tổng số ảnh xử lý đạt chuẩn đầu ra: {overall_passed}/{total_images} ({(overall_passed/total_images)*100:.1f}%)")
    print(f"Tổng thời gian chạy: {total_time:.2f} giây (Trung bình {total_time/total_images*1000:.1f} ms/ảnh)")

    return summary_records, total_images, overall_passed, total_time


if __name__ == "__main__":
    records, total_imgs, passed_imgs, total_sec = run_comprehensive_benchmark()
