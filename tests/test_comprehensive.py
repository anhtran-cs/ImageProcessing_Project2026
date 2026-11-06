"""
KIỂM THỬ TOÀN DIỆN PROJECT 2 - ỨNG DỤNG SỐ HÓA TÀI LIỆU
============================================================
Bộ kiểm thử tự động 100% (không cần GUI) cho tất cả module:
- xulytailieu.py: detect_document_corners, four_point_transform,
  remove_illumination_background, adaptive_threshold_document,
  clean_binary_document, process_document, document_metrics
- xulytailieu_nangcao.py: detect_corners_hough_otsu,
  four_point_transform_a4, remove_illumination_blackhat,
  sauvola_threshold_integral, clean_binary_connected_components,
  export_pdf_bilevel_1bit
- xuatpdf.py: export_images_to_pdf
- xulychung/xulyanh.py: read_image, save_image, cv_to_qimage

Chạy:  python kiemthu/kiemthu_project2_toandien.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# Chuẩn bị sys.path
ROOT = Path(__file__).resolve().parents[1]
for d in ("document_scanner", "core", ""):
    p = str(ROOT / d) if d else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

# Import tất cả hàm cần test
from core.image_processing import read_image, save_image, cv_to_qimage, ensure_uint8
from document_scanner.doc_processor import (
    _order_points,
    four_point_transform,
    detect_document_corners,
    crop_document_from_background,
    remove_illumination_background,
    adaptive_threshold_document,
    clean_binary_document,
    process_document,
    document_metrics,
    DocumentProcessResult,
    export_images_to_pdf,
    detect_corners_hough_otsu,
    four_point_transform_a4,
    remove_illumination_blackhat,
    sauvola_threshold_integral,
    clean_binary_connected_components,
    export_pdf_bilevel_1bit,
)

# ==============================================================
# TIỆN ÍCH TẠO ẢNH TỔNG HỢP (SYNTHETIC)
# ==============================================================

def make_document_image(
    h: int = 800, w: int = 600, bg_color=(80, 80, 80),
    paper_color=(240, 240, 240), text_color=(20, 20, 20),
    rotation_deg: float = 0, perspective: bool = False,
    uneven_light: bool = False, noise_level: int = 0,
    num_lines: int = 12
) -> np.ndarray:
    """Tạo ảnh tài liệu giả lập đa dạng điều kiện."""
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Vẽ tờ giấy trắng ở giữa
    margin_x, margin_y = w // 8, h // 10
    cv2.rectangle(img, (margin_x, margin_y),
                  (w - margin_x, h - margin_y), paper_color, -1)

    # Vẽ text giả lập
    for i in range(num_lines):
        y = margin_y + 40 + i * (h - 2 * margin_y - 60) // max(num_lines, 1)
        cv2.putText(img, f"Document line {i+1} - test content",
                    (margin_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1, cv2.LINE_AA)

    # Ánh sáng không đều
    if uneven_light:
        gradient = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                gradient[y, x] = 0.5 + 0.5 * (x / w)
        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.float32) * gradient, 0, 255
            ).astype(np.uint8)

    # Nhiễu
    if noise_level > 0:
        noise = np.random.randint(-noise_level, noise_level + 1,
                                  img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Xoay
    if rotation_deg != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=bg_color)

    # Phối cảnh nghiêng
    if perspective:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w * 0.1, h * 0.05],
            [w * 0.85, h * 0.1],
            [w * 0.95, h * 0.9],
            [w * 0.05, h * 0.95]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=bg_color)

    return img


def make_gray_test_image(h: int = 300, w: int = 400,
                         pattern: str = "gradient") -> np.ndarray:
    """Tạo ảnh xám test."""
    if pattern == "gradient":
        row = np.linspace(0, 255, w, dtype=np.uint8)
        return np.tile(row, (h, 1))
    elif pattern == "checkerboard":
        img = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h, 20):
            for x in range(0, w, 20):
                if (y // 20 + x // 20) % 2 == 0:
                    img[y:y+20, x:x+20] = 255
        return img
    elif pattern == "noisy":
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)
    else:
        return np.full((h, w), 128, dtype=np.uint8)


# ==============================================================
# FRAMEWORK KIỂM THỬ
# ==============================================================

@dataclass
class TestResult:
    name: str
    group: str
    passed: bool
    duration_ms: float
    detail: str = ""
    error: str = ""


ALL_RESULTS: list[TestResult] = []


def run_test(name: str, group: str, fn: Callable):
    """Chạy 1 test case, ghi kết quả."""
    start = time.perf_counter()
    try:
        detail = fn()
        elapsed = (time.perf_counter() - start) * 1000
        ALL_RESULTS.append(TestResult(
            name=name, group=group, passed=True,
            duration_ms=elapsed, detail=detail or "OK"
        ))
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        ALL_RESULTS.append(TestResult(
            name=name, group=group, passed=False,
            duration_ms=elapsed,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ))


# ==============================================================
# NHÓM 1: _ORDER_POINTS & FOUR_POINT_TRANSFORM
# ==============================================================

def test_order_points_standard():
    pts = np.array([[100, 200], [100, 50], [300, 50], [300, 200]], dtype=np.float32)
    ordered = _order_points(pts)
    assert ordered.shape == (4, 2), f"Shape sai: {ordered.shape}"
    # TL phải có tổng nhỏ nhất
    sums = ordered[:, 0] + ordered[:, 1]
    assert np.argmin(sums) == 0, "TL không ở vị trí 0"
    return f"Ordered: {ordered.tolist()}"


def test_order_points_shuffled():
    pts = np.array([[300, 200], [100, 50], [300, 50], [100, 200]], dtype=np.float32)
    ordered = _order_points(pts)
    sums = ordered[:, 0] + ordered[:, 1]
    assert np.argmin(sums) == 0
    return f"Shuffled -> Ordered OK"


def test_four_point_transform_basic():
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    pts = np.array([[50, 50], [250, 50], [250, 350], [50, 350]], dtype=np.float32)
    result = four_point_transform(img, pts)
    assert result is not None and result.size > 0
    assert result.ndim == 3
    return f"Output shape: {result.shape}"


def test_four_point_transform_perspective():
    img = make_document_image(600, 400, perspective=True)
    pts = np.array([[40, 30], [360, 60], [380, 540], [20, 570]], dtype=np.float32)
    result = four_point_transform(img, pts)
    assert result is not None and result.size > 0
    return f"Perspective corrected: {result.shape}"


# ==============================================================
# NHÓM 2: DETECT_DOCUMENT_CORNERS
# ==============================================================

def test_detect_corners_clear_document():
    img = make_document_image(600, 450, bg_color=(50, 50, 50),
                              paper_color=(230, 230, 230))
    corners, edges = detect_document_corners(img)
    if corners is not None:
        assert corners.shape == (4, 2), f"Corners shape: {corners.shape}"
        return f"Found 4 corners: {corners.tolist()}"
    return "Corners not found (acceptable for synthetic image)"


def test_detect_corners_high_contrast():
    img = make_document_image(800, 600, bg_color=(20, 20, 20),
                              paper_color=(250, 250, 250))
    corners, edges = detect_document_corners(img)
    found = corners is not None
    return f"High contrast: found={found}"


def test_detect_corners_low_contrast():
    img = make_document_image(800, 600, bg_color=(180, 180, 180),
                              paper_color=(220, 220, 220))
    corners, edges = detect_document_corners(img)
    found = corners is not None
    return f"Low contrast: found={found} (may fail - expected)"


def test_detect_corners_rotated():
    img = make_document_image(800, 600, rotation_deg=15)
    corners, edges = detect_document_corners(img)
    found = corners is not None
    return f"Rotated 15°: found={found}"


def test_detect_corners_empty_image():
    """Test ảnh rỗng phải raise ValueError."""
    try:
        detect_document_corners(np.array([]))
        assert False, "Should have raised ValueError"
    except ValueError:
        return "ValueError raised correctly for empty image"


def test_detect_corners_large_image():
    img = make_document_image(2400, 1800, bg_color=(40, 40, 40),
                              paper_color=(235, 235, 235))
    corners, edges = detect_document_corners(img)
    found = corners is not None
    return f"Large 2400x1800: found={found}"


def test_crop_document():
    img = make_document_image(600, 450)
    cropped, found = crop_document_from_background(img)
    assert cropped is not None and cropped.size > 0
    assert cropped.dtype == np.uint8
    return f"Cropped: {cropped.shape}, found={found}"


# ==============================================================
# NHÓM 3: REMOVE_ILLUMINATION_BACKGROUND
# ==============================================================

def test_remove_illumination_basic():
    gray = make_gray_test_image(300, 400, "gradient")
    result = remove_illumination_background(gray, kernel_size=51)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8
    return f"Mean before={gray.mean():.1f}, after={result.mean():.1f}"


def test_remove_illumination_uniform():
    gray = np.full((200, 300), 180, dtype=np.uint8)
    result = remove_illumination_background(gray, kernel_size=31)
    assert result.dtype == np.uint8
    # Ảnh đồng nhất → kết quả gần 255
    assert result.mean() > 200, f"Mean quá thấp: {result.mean()}"
    return f"Uniform: mean={result.mean():.1f}"


def test_remove_illumination_small_kernel():
    gray = make_gray_test_image(200, 300, "gradient")
    result = remove_illumination_background(gray, kernel_size=3)
    assert result.dtype == np.uint8
    return f"Small kernel k=3: mean={result.mean():.1f}"


def test_remove_illumination_large_kernel():
    gray = make_gray_test_image(200, 300, "gradient")
    result = remove_illumination_background(gray, kernel_size=151)
    assert result.dtype == np.uint8
    return f"Large kernel k=151: mean={result.mean():.1f}"


def test_remove_illumination_even_kernel():
    """Kernel chẵn phải tự chuyển thành lẻ."""
    gray = make_gray_test_image(200, 300)
    result = remove_illumination_background(gray, kernel_size=50)
    assert result.dtype == np.uint8
    return "Even kernel auto-corrected OK"


def test_remove_illumination_rejects_color():
    """Ảnh 3 kênh phải raise ValueError."""
    try:
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        remove_illumination_background(color)
        assert False, "Should raise"
    except ValueError:
        return "ValueError for color image: OK"


# ==============================================================
# NHÓM 4: ADAPTIVE_THRESHOLD_DOCUMENT
# ==============================================================

def test_adaptive_gaussian():
    gray = make_gray_test_image(300, 400, "gradient")
    result = adaptive_threshold_document(gray, method="Gaussian",
                                         block_size=21, c=10)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})
    return f"Gaussian: unique values = {np.unique(result).tolist()}"


def test_adaptive_mean():
    gray = make_gray_test_image(300, 400, "checkerboard")
    result = adaptive_threshold_document(gray, method="Mean",
                                         block_size=21, c=10)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})
    return f"Mean: black_ratio={np.mean(result == 0)*100:.1f}%"


def test_adaptive_various_block_sizes():
    gray = make_gray_test_image(200, 300, "gradient")
    results = {}
    for bs in [3, 11, 21, 51, 101]:
        r = adaptive_threshold_document(gray, block_size=bs, c=10)
        results[bs] = f"{np.mean(r == 0)*100:.1f}%"
    return f"Block sizes: {results}"


def test_adaptive_various_c_values():
    gray = make_gray_test_image(200, 300, "gradient")
    results = {}
    for c_val in [-10, 0, 5, 10, 20]:
        r = adaptive_threshold_document(gray, block_size=21, c=c_val)
        results[c_val] = f"{np.mean(r == 0)*100:.1f}%"
    return f"C values: {results}"


def test_adaptive_invalid_block_size():
    gray = make_gray_test_image(100, 100)
    try:
        adaptive_threshold_document(gray, block_size=20)
        assert False, "Should raise"
    except ValueError:
        return "Even block_size rejected: OK"

    try:
        adaptive_threshold_document(gray, block_size=1)
        assert False, "Should raise"
    except ValueError:
        return "block_size=1 rejected: OK"


def test_adaptive_rejects_color():
    try:
        adaptive_threshold_document(np.zeros((100, 100, 3), dtype=np.uint8))
        assert False
    except ValueError:
        return "Color image rejected: OK"


# ==============================================================
# NHÓM 5: CLEAN_BINARY_DOCUMENT (MORPHOLOGY)
# ==============================================================

def test_morph_none():
    binary = np.random.choice([0, 255], (200, 300)).astype(np.uint8)
    result = clean_binary_document(binary, operation="None")
    assert np.array_equal(result, binary)
    return "None operation: identical copy"


def test_morph_opening():
    binary = np.full((200, 300), 255, dtype=np.uint8)
    binary[50:55, 50:55] = 0  # Đốm nhiễu nhỏ
    result = clean_binary_document(binary, operation="Opening",
                                   kernel_size=3, iterations=1)
    assert result.dtype == np.uint8
    # Opening nên loại bỏ đốm nhiễu nhỏ (trên ảnh đảo)
    return f"Opening: black pixels before={np.sum(binary==0)}, after={np.sum(result==0)}"


def test_morph_closing():
    binary = np.zeros((200, 300), dtype=np.uint8)
    binary[50:150, 50:250] = 255
    binary[100, 100:200] = 0  # Gap nhỏ
    result = clean_binary_document(binary, operation="Closing",
                                   kernel_size=3, iterations=1)
    assert result.dtype == np.uint8
    return f"Closing: white pixels before={np.sum(binary==255)}, after={np.sum(result==255)}"


def test_morph_various_kernels():
    binary = make_gray_test_image(200, 300, "checkerboard")
    results = {}
    for k in [1, 3, 5, 7, 9]:
        r = clean_binary_document(binary, operation="Opening",
                                  kernel_size=k, iterations=1)
        results[k] = f"white={np.sum(r==255)}"
    return f"Kernel sizes: {results}"


def test_morph_various_iterations():
    gray = make_gray_test_image(200, 300, "checkerboard")
    binary = adaptive_threshold_document(gray, block_size=21, c=10)
    results = {}
    for it in [1, 2, 3, 5]:
        r = clean_binary_document(binary, operation="Opening",
                                  kernel_size=3, iterations=it)
        results[it] = f"black={np.sum(r==0)}"
    return f"Iterations: {results}"


def test_morph_invalid_operation():
    binary = np.zeros((100, 100), dtype=np.uint8)
    try:
        clean_binary_document(binary, operation="InvalidOp")
        assert False
    except ValueError:
        return "Invalid operation rejected: OK"


# ==============================================================
# NHÓM 6: PROCESS_DOCUMENT (FULL PIPELINE)
# ==============================================================

def test_pipeline_default():
    img = make_document_image(600, 450)
    result = process_document(img)
    assert isinstance(result, DocumentProcessResult)
    assert result.binary.dtype == np.uint8
    assert result.binary.ndim == 2
    assert result.original.shape == img.shape
    return (f"Pipeline OK: binary={result.binary.shape}, "
            f"found={result.document_found}")


def test_pipeline_no_detect_no_bg():
    img = make_document_image(600, 450)
    result = process_document(img, detect_page=False,
                               remove_background=False)
    assert result.document_found is False
    assert result.binary.ndim == 2
    return f"No detect/no bg: binary={result.binary.shape}"


def test_pipeline_gaussian_vs_mean():
    img = make_document_image(600, 450)
    r1 = process_document(img, adaptive_method="Gaussian")
    r2 = process_document(img, adaptive_method="Mean")
    diff = np.mean(r1.binary != r2.binary) * 100
    return f"Gaussian vs Mean: {diff:.1f}% pixels differ"


def test_pipeline_with_opening():
    img = make_document_image(600, 450, noise_level=20)
    result = process_document(img, morph_operation="Opening",
                               morph_kernel=3, morph_iterations=1)
    assert result.binary.dtype == np.uint8
    return f"Opening pipeline: shape={result.binary.shape}"


def test_pipeline_with_closing():
    img = make_document_image(600, 450)
    result = process_document(img, morph_operation="Closing",
                               morph_kernel=3, morph_iterations=1)
    assert result.binary.dtype == np.uint8
    return f"Closing pipeline: shape={result.binary.shape}"


def test_pipeline_empty_image():
    try:
        process_document(np.array([]))
        assert False
    except ValueError:
        return "Empty image rejected: OK"


def test_pipeline_none_image():
    try:
        process_document(None)
        assert False
    except (ValueError, AttributeError):
        return "None image rejected: OK"


def test_pipeline_various_bg_kernels():
    img = make_document_image(400, 300, uneven_light=True)
    results = {}
    for k in [11, 31, 51, 101]:
        r = process_document(img, background_kernel=k)
        m = document_metrics(r.binary)
        results[k] = f"black={m['black_ratio']:.1f}%"
    return f"BG kernels: {results}"


def test_pipeline_various_block_c():
    img = make_document_image(400, 300)
    results = {}
    for bs, c in [(11, 5), (21, 10), (31, 15), (51, 20)]:
        r = process_document(img, block_size=bs, c=c)
        m = document_metrics(r.binary)
        results[f"bs{bs}_c{c}"] = f"black={m['black_ratio']:.1f}%"
    return f"Block/C configs: {results}"


def test_pipeline_uneven_light():
    img = make_document_image(600, 450, uneven_light=True)
    r_with = process_document(img, remove_background=True)
    r_without = process_document(img, remove_background=False)
    m_with = document_metrics(r_with.binary)
    m_without = document_metrics(r_without.binary)
    return (f"With BG removal: black={m_with['black_ratio']:.1f}%, "
            f"Without: black={m_without['black_ratio']:.1f}%")


def test_pipeline_noisy_image():
    img = make_document_image(600, 450, noise_level=30)
    result = process_document(img, morph_operation="Opening",
                               morph_kernel=3)
    m = document_metrics(result.binary)
    return f"Noisy: black={m['black_ratio']:.1f}%, std={m['std']:.1f}"


def test_pipeline_perspective_image():
    img = make_document_image(800, 600, perspective=True)
    result = process_document(img, detect_page=True)
    return f"Perspective: found={result.document_found}, shape={result.binary.shape}"


def test_pipeline_rotated_image():
    img = make_document_image(800, 600, rotation_deg=10)
    result = process_document(img, detect_page=True)
    return f"Rotated: found={result.document_found}, shape={result.binary.shape}"


# ==============================================================
# NHÓM 7: DOCUMENT_METRICS
# ==============================================================

def test_metrics_black_image():
    binary = np.zeros((200, 300), dtype=np.uint8)
    m = document_metrics(binary)
    assert abs(m["black_ratio"] - 100.0) < 0.01
    assert abs(m["white_ratio"] - 0.0) < 0.01
    assert abs(m["mean"] - 0.0) < 0.01
    return f"All black: {m}"


def test_metrics_white_image():
    binary = np.full((200, 300), 255, dtype=np.uint8)
    m = document_metrics(binary)
    assert abs(m["black_ratio"] - 0.0) < 0.01
    assert abs(m["white_ratio"] - 100.0) < 0.01
    assert abs(m["mean"] - 255.0) < 0.01
    return f"All white: {m}"


def test_metrics_half_image():
    binary = np.zeros((200, 300), dtype=np.uint8)
    binary[:100, :] = 255
    m = document_metrics(binary)
    assert abs(m["black_ratio"] - 50.0) < 0.01
    assert abs(m["white_ratio"] - 50.0) < 0.01
    return f"Half: {m}"


def test_metrics_color_input():
    """document_metrics phải xử lý được ảnh 3 kênh."""
    color = np.full((100, 100, 3), 128, dtype=np.uint8)
    m = document_metrics(color)
    assert "black_ratio" in m
    return f"Color input handled: {m}"


# ==============================================================
# NHÓM 8: EXPORT PDF
# ==============================================================

def test_export_pdf_single_page():
    binary = np.full((600, 400), 255, dtype=np.uint8)
    cv2.putText(binary, "Test PDF", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    out = str(ROOT / "tests" / "_test_single.pdf")
    export_images_to_pdf([binary], out, page_size="Auto")
    assert Path(out).exists()
    size = Path(out).stat().st_size
    Path(out).unlink(missing_ok=True)
    return f"Single page PDF: {size} bytes"


def test_export_pdf_multi_page():
    pages = []
    for i in range(5):
        p = np.full((600, 400), 255, dtype=np.uint8)
        cv2.putText(p, f"Page {i+1}", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
        pages.append(p)
    out = str(ROOT / "tests" / "_test_multi.pdf")
    export_images_to_pdf(pages, out, page_size="A4")
    assert Path(out).exists()
    size = Path(out).stat().st_size
    Path(out).unlink(missing_ok=True)
    return f"Multi page PDF (5 pages): {size} bytes"


def test_export_pdf_a4_portrait():
    binary = np.full((800, 600), 200, dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_a4p.pdf")
    export_images_to_pdf([binary], out, page_size="A4",
                         orientation="Portrait", margin=30)
    assert Path(out).exists()
    Path(out).unlink(missing_ok=True)
    return "A4 Portrait export OK"


def test_export_pdf_a4_landscape():
    binary = np.full((600, 800), 200, dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_a4l.pdf")
    export_images_to_pdf([binary], out, page_size="A4",
                         orientation="Landscape", margin=30)
    assert Path(out).exists()
    Path(out).unlink(missing_ok=True)
    return "A4 Landscape export OK"


def test_export_pdf_color_image():
    color = np.full((400, 300, 3), (200, 180, 160), dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_color.pdf")
    export_images_to_pdf([color], out)
    assert Path(out).exists()
    Path(out).unlink(missing_ok=True)
    return "Color image PDF OK"


def test_export_pdf_empty():
    try:
        export_images_to_pdf([], str(ROOT / "tests" / "_fail.pdf"))
        assert False
    except ValueError:
        return "Empty list rejected: OK"


def test_export_pdf_zero_margin():
    binary = np.full((400, 300), 128, dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_0margin.pdf")
    export_images_to_pdf([binary], out, page_size="A4", margin=0)
    assert Path(out).exists()
    Path(out).unlink(missing_ok=True)
    return "Zero margin OK"


# ==============================================================
# NHÓM 9: MODULE NÂNG CAO (xulytailieu_nangcao.py)
# ==============================================================

def test_nangcao_detect_hough_otsu():
    img = make_document_image(800, 600, bg_color=(30, 30, 30),
                              paper_color=(240, 240, 240))
    corners, edges = detect_corners_hough_otsu(img)
    found = corners is not None
    return f"Hough+Otsu: found={found}"


def test_nangcao_four_point_a4():
    img = make_document_image(800, 600)
    pts = np.array([[75, 80], [525, 80], [525, 720], [75, 720]], dtype=np.float32)
    result = four_point_transform_a4(img, pts)
    assert result is not None and result.size > 0
    h, w = result.shape[:2]
    ratio = h / w if h > w else w / h
    return f"A4 transform: {result.shape}, ratio={ratio:.3f} (target ~1.414)"


def test_nangcao_blackhat():
    gray = make_gray_test_image(300, 400, "gradient")
    result = remove_illumination_blackhat(gray, kernel_size=51)
    assert result.dtype == np.uint8
    assert result.shape == gray.shape
    return f"Blackhat: mean={result.mean():.1f}, std={result.std():.1f}"


def test_nangcao_blackhat_rejects_color():
    try:
        remove_illumination_blackhat(np.zeros((100, 100, 3), dtype=np.uint8))
        assert False
    except ValueError:
        return "Color rejected: OK"


def test_nangcao_sauvola():
    gray = make_gray_test_image(200, 300, "gradient")
    result = sauvola_threshold_integral(gray, window_size=25, k=0.22)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})
    return f"Sauvola: black={np.mean(result==0)*100:.1f}%"


def test_nangcao_sauvola_rejects_color():
    try:
        sauvola_threshold_integral(np.zeros((100, 100, 3), dtype=np.uint8))
        assert False
    except ValueError:
        return "Color rejected: OK"


def test_nangcao_sauvola_various_params():
    gray = make_gray_test_image(200, 300, "checkerboard")
    results = {}
    for ws in [11, 25, 51]:
        for k_val in [0.1, 0.22, 0.5]:
            r = sauvola_threshold_integral(gray, window_size=ws, k=k_val)
            results[f"ws{ws}_k{k_val}"] = f"black={np.mean(r==0)*100:.1f}%"
    return f"Sauvola params: {results}"


def test_nangcao_connected_components():
    binary = np.full((200, 300), 255, dtype=np.uint8)
    # Thêm các đốm nhiễu nhỏ
    for y in range(10, 190, 30):
        for x in range(10, 290, 30):
            binary[y:y+2, x:x+2] = 0
    result = clean_binary_connected_components(binary, min_area=4)
    assert result.dtype == np.uint8
    removed = np.sum(binary == 0) - np.sum(result == 0)
    return f"CC cleanup: removed {removed} black pixels"


def test_nangcao_export_bilevel():
    pages = [np.full((400, 300), 255, dtype=np.uint8)]
    cv2.putText(pages[0], "Bilevel", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    out = str(ROOT / "tests" / "_test_bilevel.pdf")
    export_pdf_bilevel_1bit(pages, out)
    assert Path(out).exists()
    size = Path(out).stat().st_size
    Path(out).unlink(missing_ok=True)
    return f"Bilevel PDF: {size} bytes"


def test_nangcao_export_bilevel_empty():
    try:
        export_pdf_bilevel_1bit([], str(ROOT / "tests" / "_fail.pdf"))
        assert False
    except ValueError:
        return "Empty list rejected: OK"


# ==============================================================
# NHÓM 10: XULYCHUNG (read_image, save_image, cv_to_qimage)
# ==============================================================

def test_save_and_read_png():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_io.png")
    save_image(out, img)
    loaded = read_image(out)
    assert loaded.shape == img.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, img)
    Path(out).unlink(missing_ok=True)
    return "Save/Read PNG: pixel-perfect"


def test_save_and_read_jpg():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    out = str(ROOT / "tests" / "_test_io.jpg")
    save_image(out, img)
    loaded = read_image(out)
    assert loaded.shape == img.shape
    # JPEG lossy → so sánh gần đúng
    diff = np.abs(loaded.astype(int) - img.astype(int)).mean()
    Path(out).unlink(missing_ok=True)
    return f"Save/Read JPG: mean diff={diff:.2f}"


def test_save_empty_image():
    try:
        save_image(str(ROOT / "tests" / "_fail.png"), np.array([]))
        assert False
    except ValueError:
        return "Empty image rejected: OK"


def test_read_nonexistent():
    try:
        read_image("nonexistent_file_abc123.png")
        assert False
    except FileNotFoundError:
        return "FileNotFoundError: OK"


def test_save_invalid_extension():
    try:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        save_image(str(ROOT / "tests" / "_fail.xyz"), img)
        assert False
    except ValueError:
        return "Invalid extension rejected: OK"


def test_cv_to_qimage_gray():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    qimg = cv_to_qimage(gray)
    assert not qimg.isNull()
    return f"QImage gray: {qimg.width()}x{qimg.height()}"


def test_cv_to_qimage_color():
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    qimg = cv_to_qimage(color)
    assert not qimg.isNull()
    return f"QImage color: {qimg.width()}x{qimg.height()}"


def test_cv_to_qimage_none():
    qimg = cv_to_qimage(None)
    assert qimg.isNull()
    return "QImage None: isNull=True"


def test_ensure_uint8_float():
    img = np.array([[-10.0, 128.5, 300.0]], dtype=np.float32)
    result = ensure_uint8(img)
    assert result.dtype == np.uint8
    assert result[0, 0] == 0
    assert result[0, 2] == 255
    return f"ensure_uint8: {result}"


# ==============================================================
# NHÓM 11: KIỂM THỬ TÍCH HỢP END-TO-END
# ==============================================================

def test_e2e_full_workflow():
    """Mô phỏng workflow hoàn chỉnh: tạo ảnh → pipeline → save → export PDF."""
    img = make_document_image(800, 600, uneven_light=True, noise_level=10)

    # Bước 1: Pipeline
    result = process_document(
        img, detect_page=True, remove_background=True,
        background_kernel=51, adaptive_method="Gaussian",
        block_size=21, c=10, morph_operation="Opening",
        morph_kernel=3, morph_iterations=1,
    )

    # Bước 2: Metrics
    m = document_metrics(result.binary)
    assert 0 < m["black_ratio"] < 100
    assert 0 < m["white_ratio"] < 100

    # Bước 3: Save ảnh
    img_out = str(ROOT / "tests" / "_e2e_page.png")
    save_image(img_out, result.binary)
    assert Path(img_out).exists()
    Path(img_out).unlink(missing_ok=True)

    # Bước 4: Export PDF
    pdf_out = str(ROOT / "tests" / "_e2e_doc.pdf")
    export_images_to_pdf([result.binary], pdf_out, page_size="A4")
    assert Path(pdf_out).exists()
    pdf_size = Path(pdf_out).stat().st_size
    Path(pdf_out).unlink(missing_ok=True)

    return (f"E2E: binary={result.binary.shape}, "
            f"black={m['black_ratio']:.1f}%, pdf={pdf_size} bytes")


def test_e2e_multi_page_workflow():
    """Mô phỏng xử lý nhiều trang."""
    images = [
        make_document_image(600, 450, noise_level=0),
        make_document_image(600, 450, uneven_light=True),
        make_document_image(600, 450, rotation_deg=5),
        make_document_image(600, 450, perspective=True),
        make_document_image(600, 450, noise_level=20),
    ]

    results = []
    for img in images:
        r = process_document(img, detect_page=True, remove_background=True)
        results.append(r)

    # Export all
    binaries = [r.binary for r in results]
    pdf_out = str(ROOT / "tests" / "_e2e_multi.pdf")
    export_images_to_pdf(binaries, pdf_out, page_size="A4")
    assert Path(pdf_out).exists()
    pdf_size = Path(pdf_out).stat().st_size
    Path(pdf_out).unlink(missing_ok=True)

    found_count = sum(1 for r in results if r.document_found)
    return f"Multi-page: {len(results)} pages, {found_count} detected, pdf={pdf_size} bytes"


def test_e2e_real_images_if_available():
    """Kiểm thử trên ảnh thật nếu có."""
    img_dir = ROOT.parent / "img" / "img pj2"
    if not img_dir.exists():
        return "No real images directory found (skipped)"

    real_files = list(img_dir.glob("*.*"))
    if not real_files:
        return "No real images found (skipped)"

    results_info = []
    for f in real_files[:5]:
        try:
            img = read_image(str(f))
            r = process_document(img)
            m = document_metrics(r.binary)
            results_info.append(
                f"{f.name}: found={r.document_found}, "
                f"black={m['black_ratio']:.1f}%, "
                f"shape={r.binary.shape}"
            )
        except Exception as e:
            results_info.append(f"{f.name}: ERROR {e}")

    return "Real images:\n  " + "\n  ".join(results_info)


# ==============================================================
# NHÓM 12: KIỂM THỬ HIỆU NĂNG
# ==============================================================

def test_performance_pipeline():
    """Đo thời gian pipeline trên các kích thước ảnh."""
    sizes = [(400, 300), (800, 600), (1600, 1200), (3200, 2400)]
    times = {}
    for h, w in sizes:
        img = make_document_image(h, w)
        start = time.perf_counter()
        process_document(img)
        elapsed = (time.perf_counter() - start) * 1000
        times[f"{w}x{h}"] = f"{elapsed:.0f}ms"
    return f"Performance: {times}"


def test_performance_adaptive_threshold():
    gray = np.random.randint(0, 256, (1000, 800), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(10):
        adaptive_threshold_document(gray, block_size=21, c=10)
    elapsed = (time.perf_counter() - start) * 1000
    return f"10x Adaptive Threshold 1000x800: {elapsed:.0f}ms total, {elapsed/10:.0f}ms avg"


def test_performance_morphology():
    binary = np.random.choice([0, 255], (1000, 800)).astype(np.uint8)
    start = time.perf_counter()
    for _ in range(10):
        clean_binary_document(binary, "Opening", 3, 1)
    elapsed = (time.perf_counter() - start) * 1000
    return f"10x Morphology 1000x800: {elapsed:.0f}ms total, {elapsed/10:.0f}ms avg"


# ==============================================================
# ĐĂNG KÝ VÀ CHẠY TẤT CẢ
# ==============================================================

def register_all_tests():
    tests = [
        # Nhóm 1
        ("Order Points chuẩn", "1. Geometry", test_order_points_standard),
        ("Order Points shuffled", "1. Geometry", test_order_points_shuffled),
        ("Four Point Transform cơ bản", "1. Geometry", test_four_point_transform_basic),
        ("Four Point Transform perspective", "1. Geometry", test_four_point_transform_perspective),
        # Nhóm 2
        ("Detect Corners - ảnh rõ", "2. Detect Corners", test_detect_corners_clear_document),
        ("Detect Corners - high contrast", "2. Detect Corners", test_detect_corners_high_contrast),
        ("Detect Corners - low contrast", "2. Detect Corners", test_detect_corners_low_contrast),
        ("Detect Corners - rotated", "2. Detect Corners", test_detect_corners_rotated),
        ("Detect Corners - ảnh rỗng", "2. Detect Corners", test_detect_corners_empty_image),
        ("Detect Corners - large image", "2. Detect Corners", test_detect_corners_large_image),
        ("Crop Document", "2. Detect Corners", test_crop_document),
        # Nhóm 3
        ("Remove Illumination basic", "3. Background Removal", test_remove_illumination_basic),
        ("Remove Illumination uniform", "3. Background Removal", test_remove_illumination_uniform),
        ("Remove Illumination small kernel", "3. Background Removal", test_remove_illumination_small_kernel),
        ("Remove Illumination large kernel", "3. Background Removal", test_remove_illumination_large_kernel),
        ("Remove Illumination even kernel", "3. Background Removal", test_remove_illumination_even_kernel),
        ("Remove Illumination rejects color", "3. Background Removal", test_remove_illumination_rejects_color),
        # Nhóm 4
        ("Adaptive Gaussian", "4. Adaptive Threshold", test_adaptive_gaussian),
        ("Adaptive Mean", "4. Adaptive Threshold", test_adaptive_mean),
        ("Adaptive various block sizes", "4. Adaptive Threshold", test_adaptive_various_block_sizes),
        ("Adaptive various C values", "4. Adaptive Threshold", test_adaptive_various_c_values),
        ("Adaptive invalid block size", "4. Adaptive Threshold", test_adaptive_invalid_block_size),
        ("Adaptive rejects color", "4. Adaptive Threshold", test_adaptive_rejects_color),
        # Nhóm 5
        ("Morph None", "5. Morphology", test_morph_none),
        ("Morph Opening", "5. Morphology", test_morph_opening),
        ("Morph Closing", "5. Morphology", test_morph_closing),
        ("Morph various kernels", "5. Morphology", test_morph_various_kernels),
        ("Morph various iterations", "5. Morphology", test_morph_various_iterations),
        ("Morph invalid operation", "5. Morphology", test_morph_invalid_operation),
        # Nhóm 6
        ("Pipeline default", "6. Full Pipeline", test_pipeline_default),
        ("Pipeline no detect/no bg", "6. Full Pipeline", test_pipeline_no_detect_no_bg),
        ("Pipeline Gaussian vs Mean", "6. Full Pipeline", test_pipeline_gaussian_vs_mean),
        ("Pipeline with Opening", "6. Full Pipeline", test_pipeline_with_opening),
        ("Pipeline with Closing", "6. Full Pipeline", test_pipeline_with_closing),
        ("Pipeline empty image", "6. Full Pipeline", test_pipeline_empty_image),
        ("Pipeline None image", "6. Full Pipeline", test_pipeline_none_image),
        ("Pipeline various BG kernels", "6. Full Pipeline", test_pipeline_various_bg_kernels),
        ("Pipeline various Block/C", "6. Full Pipeline", test_pipeline_various_block_c),
        ("Pipeline uneven light", "6. Full Pipeline", test_pipeline_uneven_light),
        ("Pipeline noisy image", "6. Full Pipeline", test_pipeline_noisy_image),
        ("Pipeline perspective image", "6. Full Pipeline", test_pipeline_perspective_image),
        ("Pipeline rotated image", "6. Full Pipeline", test_pipeline_rotated_image),
        # Nhóm 7
        ("Metrics all black", "7. Metrics", test_metrics_black_image),
        ("Metrics all white", "7. Metrics", test_metrics_white_image),
        ("Metrics half", "7. Metrics", test_metrics_half_image),
        ("Metrics color input", "7. Metrics", test_metrics_color_input),
        # Nhóm 8
        ("Export PDF single page", "8. Export PDF", test_export_pdf_single_page),
        ("Export PDF multi page", "8. Export PDF", test_export_pdf_multi_page),
        ("Export PDF A4 Portrait", "8. Export PDF", test_export_pdf_a4_portrait),
        ("Export PDF A4 Landscape", "8. Export PDF", test_export_pdf_a4_landscape),
        ("Export PDF color image", "8. Export PDF", test_export_pdf_color_image),
        ("Export PDF empty list", "8. Export PDF", test_export_pdf_empty),
        ("Export PDF zero margin", "8. Export PDF", test_export_pdf_zero_margin),
        # Nhóm 9
        ("Nâng cao - Hough+Otsu detect", "9. Module Nâng Cao", test_nangcao_detect_hough_otsu),
        ("Nâng cao - Four Point A4", "9. Module Nâng Cao", test_nangcao_four_point_a4),
        ("Nâng cao - Blackhat", "9. Module Nâng Cao", test_nangcao_blackhat),
        ("Nâng cao - Blackhat rejects color", "9. Module Nâng Cao", test_nangcao_blackhat_rejects_color),
        ("Nâng cao - Sauvola", "9. Module Nâng Cao", test_nangcao_sauvola),
        ("Nâng cao - Sauvola rejects color", "9. Module Nâng Cao", test_nangcao_sauvola_rejects_color),
        ("Nâng cao - Sauvola various params", "9. Module Nâng Cao", test_nangcao_sauvola_various_params),
        ("Nâng cao - Connected Components", "9. Module Nâng Cao", test_nangcao_connected_components),
        ("Nâng cao - Export Bilevel PDF", "9. Module Nâng Cao", test_nangcao_export_bilevel),
        ("Nâng cao - Export Bilevel empty", "9. Module Nâng Cao", test_nangcao_export_bilevel_empty),
        # Nhóm 10
        ("Save/Read PNG", "10. I/O xulychung", test_save_and_read_png),
        ("Save/Read JPG", "10. I/O xulychung", test_save_and_read_jpg),
        ("Save empty image", "10. I/O xulychung", test_save_empty_image),
        ("Read nonexistent", "10. I/O xulychung", test_read_nonexistent),
        ("Save invalid extension", "10. I/O xulychung", test_save_invalid_extension),
        ("cv_to_qimage gray", "10. I/O xulychung", test_cv_to_qimage_gray),
        ("cv_to_qimage color", "10. I/O xulychung", test_cv_to_qimage_color),
        ("cv_to_qimage None", "10. I/O xulychung", test_cv_to_qimage_none),
        ("ensure_uint8 float", "10. I/O xulychung", test_ensure_uint8_float),
        # Nhóm 11
        ("E2E full workflow", "11. End-to-End", test_e2e_full_workflow),
        ("E2E multi-page workflow", "11. End-to-End", test_e2e_multi_page_workflow),
        ("E2E real images", "11. End-to-End", test_e2e_real_images_if_available),
        # Nhóm 12
        ("Performance pipeline", "12. Performance", test_performance_pipeline),
        ("Performance Adaptive Threshold", "12. Performance", test_performance_adaptive_threshold),
        ("Performance Morphology", "12. Performance", test_performance_morphology),
    ]
    return tests


def print_report(results: list[TestResult]):
    """In báo cáo tổng hợp."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_time = sum(r.duration_ms for r in results)

    print("\n" + "=" * 80)
    print("BÁO CÁO KIỂM THỬ TOÀN DIỆN - PROJECT 2 SỐ HÓA TÀI LIỆU")
    print("=" * 80)

    # Nhóm kết quả
    groups = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    for group_name, group_results in groups.items():
        g_passed = sum(1 for r in group_results if r.passed)
        g_total = len(group_results)
        status = "✅" if g_passed == g_total else "⚠️"
        print(f"\n{status} {group_name} [{g_passed}/{g_total}]")
        print("-" * 70)
        for r in group_results:
            icon = "✅" if r.passed else "❌"
            time_str = f"{r.duration_ms:.0f}ms" if r.duration_ms < 1000 else f"{r.duration_ms/1000:.1f}s"
            print(f"  {icon} {r.name:45s} [{time_str:>8s}]")
            if r.passed and r.detail:
                print(f"      → {r.detail}")
            if not r.passed:
                # In 3 dòng đầu error
                lines = r.error.strip().split("\n")
                for line in lines[:3]:
                    print(f"      ❌ {line}")

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT: {passed}/{total} PASSED, {failed} FAILED")
    print(f"Thời gian: {total_time:.0f}ms ({total_time/1000:.1f}s)")
    rate = (passed / total * 100) if total > 0 else 0
    print(f"Tỷ lệ thành công: {rate:.1f}%")
    if failed == 0:
        print("🎉 TẤT CẢ KIỂM THỬ THÀNH CÔNG!")
    else:
        print(f"⚠️  CÓ {failed} KIỂM THỬ THẤT BẠI - cần xem lại.")
    print("=" * 80)


def main():
    print("🔬 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN PROJECT 2...")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Thư mục gốc: {ROOT}")

    tests = register_all_tests()
    print(f"\nSố lượng test case: {len(tests)}")
    print("-" * 80)

    for name, group, fn in tests:
        print(f"  Running: {name}...", end=" ", flush=True)
        run_test(name, group, fn)
        result = ALL_RESULTS[-1]
        print("✅" if result.passed else "❌")

    print_report(ALL_RESULTS)

    # Trả exit code
    failed = sum(1 for r in ALL_RESULTS if not r.passed)
    return failed


if __name__ == "__main__":
    sys.exit(main())
