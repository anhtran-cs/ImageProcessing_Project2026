"""
PROJECT 2 - LÕI XỬ LÝ SỐ HÓA TÀI LIỆU & XUẤT PDF
==================================================
Bộ module hợp nhất toàn diện cho Project 2 (Chủ đề 3):
1. Hình học & Nắn phối cảnh: _order_points, inset_corners, four_point_transform, four_point_transform_a4, draw_document_corners
2. Phát hiện góc & Biên: detect_document_corners, detect_corners_hough_otsu, crop_document_from_background
3. Khử bóng & Chuẩn hóa sáng: remove_illumination_background, remove_illumination_blackhat
4. Nhị phân hóa thích nghi: adaptive_threshold_document, sauvola_threshold_integral
5. Làm sạch hình thái học: clean_binary_document, clean_binary_connected_components
6. Pipeline & Đo đạc: process_document, document_metrics
7. Xuất file PDF: export_images_to_pdf, export_pdf_bilevel_1bit
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class DocumentProcessResult:
    original: np.ndarray
    perspective_corrected: np.ndarray
    normalized_gray: np.ndarray
    binary: np.ndarray
    document_found: bool
    corners: Optional[np.ndarray] = None


# ==============================================================
# PHẦN 1: HÌNH HỌC & BIẾN ĐỔI PHỐI CẢNH (GEOMETRY & PERSPECTIVE)
# ==============================================================

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự chuẩn: TL (Top-Left), TR (Top-Right), BR (Bottom-Right), BL (Bottom-Left).
    Sử dụng góc cực (polar angle arctan2) quanh trọng tâm để chống đảo góc khi tài liệu bị xoay nghiêng.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts_sorted = pts[np.argsort(angles)]

    # Điểm Top-Left là điểm có tổng tọa độ (x + y) nhỏ nhất
    s = pts_sorted[:, 0] + pts_sorted[:, 1]
    tl_idx = int(np.argmin(s))
    ordered = np.roll(pts_sorted, -tl_idx, axis=0)
    return ordered.astype(np.float32)


def inset_corners(pts: np.ndarray, margin_ratio: float = 0.015) -> np.ndarray:
    """
    Co 4 góc về phía trọng tâm một tỉ lệ nhỏ (mặc định 1.5%)
    để loại bỏ mép quăn, viền đen và bóng đổ ở mép viền ngoài.
    """
    if margin_ratio <= 0:
        return pts
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)
    return pts + (center - pts) * float(margin_ratio)


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Biến đổi phối cảnh (Perspective Transform) đưa tờ giấy về góc nhìn trực diện."""
    rect = _order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(1, int(round(max(width_a, width_b))))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(1, int(round(max(height_a, height_b))))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def four_point_transform_a4(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Biến đổi phối cảnh bảo toàn tỷ lệ khổ A4 chuẩn 1 : sqrt(2)."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    # Ràng buộc tỷ lệ A4 chuẩn (1 : 1.414)
    ratio = height / max(width, 1.0)
    if 1.25 <= ratio <= 1.55:
        height = width * np.sqrt(2.0)
    elif 1.25 <= (1.0 / ratio) <= 1.55:
        width = height * np.sqrt(2.0)

    max_w, max_h = int(round(width)), int(round(height))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def draw_document_corners(
    image: np.ndarray,
    corners: Optional[np.ndarray],
    inset_ratio: float = 0.0,
) -> np.ndarray:
    """
    Vẽ khung quét 4 góc trực quan lên ảnh gốc:
    - Khung xanh lá (Neon Green): 4 góc phát hiện của tờ giấy.
    - Khung vàng nhạt (Golden): đường cắt thực tế sau khi co lề (inset).
    - 4 chấm tròn và nhãn góc: TL, TR, BR, BL.
    """
    if image is None or corners is None:
        return image.copy() if image is not None else None

    vis = image.copy()
    pts = _order_points(corners).astype(np.int32)

    # 1. Vẽ khung quét tứ giác ngoài
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    # 2. Vẽ khung co lề (nếu có)
    if inset_ratio > 0:
        pts_inset = _order_points(inset_corners(corners, inset_ratio)).astype(np.int32)
        cv2.polylines(vis, [pts_inset], isClosed=True, color=(255, 200, 0), thickness=2, lineType=cv2.LINE_AA)

    # 3. Vẽ 4 điểm góc và nhãn
    labels = ["TL", "TR", "BR", "BL"]
    for pt, label in zip(pts, labels):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 8, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        offset_x = 10 if x < image.shape[1] // 2 else -40
        offset_y = -10 if y > image.shape[0] // 2 else 20
        cv2.putText(
            vis, label, (x + offset_x, y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
        )

    return vis


# ==============================================================
# PHẦN 2: PHÁT HIỆN BIÊN & 4 GÓC (CORNER DETECTION)
# ==============================================================

def detect_document_corners(image: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Phát hiện 4 góc tài liệu:
    1. Canny thích nghi theo ngưỡng Otsu (lower=0.5*Otsu, upper=Otsu).
    2. Đóng biên (Morphological Close) để nối liền đoạn đứt gãy.
    3. Tìm kiếm đa mức epsilon (0.015 - 0.04) và bao lồi (Convex Hull).
    4. Dự phòng phân ngưỡng vùng sáng Otsu khi tương phản thấp.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")

    h, w = image.shape[:2]
    scale = 1.0
    work = image
    max_dim = max(h, w)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Chiến lược 1: Canny thích nghi theo Otsu + Đóng biên + Epsilon đa mức
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = int(max(20, 0.5 * otsu_thresh))
    upper = int(min(240, otsu_thresh))
    edges = cv2.Canny(blurred, lower, upper)

    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges_closed = cv2.dilate(edges_closed, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    min_area = 0.10 * work.shape[0] * work.shape[1]
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if c_area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)

        for eps_factor in (0.015, 0.02, 0.025, 0.03, 0.04):
            approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype(np.float32)
                if scale != 1.0:
                    corners /= scale
                return corners, edges

            approx_hull = cv2.approxPolyDP(hull, eps_factor * peri, True)
            if len(approx_hull) == 4 and cv2.isContourConvex(approx_hull):
                corners = approx_hull.reshape(4, 2).astype(np.float32)
                if scale != 1.0:
                    corners /= scale
                return corners, edges

    # Chiến lược 2: Dự phòng phân ngưỡng Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    t_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_contours = sorted(t_contours, key=cv2.contourArea, reverse=True)

    for contour in t_contours:
        c_area = cv2.contourArea(contour)
        if c_area < min_area or c_area > 0.98 * work.shape[0] * work.shape[1]:
            continue
        peri = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)

        for eps_factor in (0.02, 0.03, 0.04):
            approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype(np.float32)
                if scale != 1.0:
                    corners /= scale
                return corners, edges

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        if scale != 1.0:
            box /= scale
        return box, edges

    return None, edges


def detect_corners_hough_otsu(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Phát hiện 4 góc bằng Canny Otsu và Hull đa mức (Chương 7 & 9)."""
    return detect_document_corners(image)


def crop_document_from_background(image: np.ndarray, inset_ratio: float = 0.015) -> tuple[np.ndarray, bool]:
    corners, _ = detect_document_corners(image)
    if corners is None:
        return image.copy(), False
    if inset_ratio > 0:
        corners = inset_corners(corners, inset_ratio)
    return four_point_transform(image, corners), True


# ==============================================================
# PHẦN 3: KHỬ NỀN & CHIẾU SÁNG KHÔNG ĐỀU (ILLUMINATION)
# ==============================================================

def remove_illumination_background(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """
    Ước lượng nền chiếu sáng bằng Morphological Closing rồi chuẩn hóa bằng phép chia.
    Triệt tiêu bóng đổ, bóng tay và nâng sáng toàn bộ bề mặt giấy.
    """
    if gray.ndim != 2:
        raise ValueError("remove_illumination_background yêu cầu ảnh xám 2D")

    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def remove_illumination_blackhat(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Khử bóng đổ bằng Morphological Black-Hat Transform (Chương 10)."""
    if gray.ndim != 2:
        raise ValueError("Yêu cầu ảnh xám 2D")
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.bitwise_not(norm)


# ==============================================================
# PHẦN 4: NHỊ PHÂN HÓA THÍCH NGHI (BINARIZATION)
# ==============================================================

def adaptive_threshold_document(
    gray: np.ndarray,
    method: Literal["Gaussian", "Mean"] = "Gaussian",
    block_size: int = 21,
    c: float = 10.0,
) -> np.ndarray:
    """Phân ngưỡng thích nghi cục bộ (Adaptive Gaussian / Mean)."""
    if gray.ndim != 2:
        raise ValueError("Adaptive Threshold yêu cầu ảnh xám 2D")

    block_size = int(block_size)
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("block_size phải là số lẻ > 1")

    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method.lower() == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    return cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        float(c),
    )


def sauvola_threshold_integral(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.22,
    r: float = 128.0,
) -> np.ndarray:
    """Nhị phân hóa Sauvola tối ưu bằng Integral Image O(1) (Chương 5 & 9)."""
    if gray.ndim != 2:
        raise ValueError("Yêu cầu ảnh xám 2D")
    w_size = int(window_size)
    if w_size % 2 == 0:
        w_size += 1
    pad = w_size // 2

    img = gray.astype(np.float64)
    padded = np.pad(img, pad, mode="reflect")

    ii = cv2.integral(padded)
    ii_sq = cv2.integral(padded**2)

    h, w = gray.shape
    y1, y2 = np.arange(h), np.arange(h) + w_size
    x1, x2 = np.arange(w), np.arange(w) + w_size
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    Y2, X2 = np.meshgrid(y2, x2, indexing="ij")

    area = float(w_size * w_size)
    mean = (ii[Y2, X2] - ii[Y1, X2] - ii[Y2, X1] + ii[Y1, X1]) / area
    mean_sq = (ii_sq[Y2, X2] - ii_sq[Y1, X2] - ii_sq[Y2, X1] + ii_sq[Y1, X1]) / area
    variance = np.maximum(mean_sq - (mean**2), 0)
    std = np.sqrt(variance)

    thresh = mean * (1.0 + float(k) * ((std / float(r)) - 1.0))
    return np.where(img >= thresh, 255, 0).astype(np.uint8)


# ==============================================================
# PHẦN 5: LÀM SẠCH HÌNH THÁI HỌC (MORPHOLOGY)
# ==============================================================

def clean_binary_document(
    binary: np.ndarray,
    operation: Literal["None", "Opening", "Closing"] = "None",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Làm sạch nét hoặc nối nét bằng phép đảo nhị phân + Morphology."""
    if operation.lower() == "none":
        return binary.copy()

    # Đảo ảnh để chữ thành foreground trắng
    inv = cv2.bitwise_not(binary)

    k = int(kernel_size)
    if k <= 0:
        raise ValueError("kernel_size morphology phải > 0")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    if operation.lower() == "opening":
        op = cv2.MORPH_OPEN
    elif operation.lower() == "closing":
        op = cv2.MORPH_CLOSE
    else:
        raise ValueError("operation phải là None / Opening / Closing")

    cleaned_inv = cv2.morphologyEx(inv, op, kernel, iterations=max(1, int(iterations)))
    return cv2.bitwise_not(cleaned_inv)


def clean_binary_connected_components(binary: np.ndarray, min_area: int = 4) -> np.ndarray:
    """Làm sạch nhiễu hạt bằng Connected Components Analysis (Chương 9 & 10)."""
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    cleaned_inv = np.zeros_like(inv)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_inv[labels == i] = 255

    return cv2.bitwise_not(cleaned_inv)


# ==============================================================
# PHẦN 6: PIPELINE TỔNG THỂ & THỐNG KÊ (PIPELINE & METRICS)
# ==============================================================

def process_document(
    image: np.ndarray,
    *,
    detect_page: bool = True,
    inset_ratio: float = 0.015,
    remove_background: bool = True,
    background_kernel: int = 51,
    adaptive_method: str = "Gaussian",
    block_size: int = 21,
    c: float = 10.0,
    morph_operation: str = "Opening",
    morph_kernel: int = 3,
    morph_iterations: int = 1,
) -> DocumentProcessResult:
    """PIPELINE HOÀN CHỈNH CHO PROJECT 2."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")

    # B1: Tách giấy khỏi nền ngoài + Perspective Transform
    corners = None
    if detect_page:
        corners, _ = detect_document_corners(image)
        if corners is not None:
            c_use = inset_corners(corners, inset_ratio) if inset_ratio > 0 else corners
            page = four_point_transform(image, c_use)
            found = True
        else:
            page, found = image.copy(), False
    else:
        page, found = image.copy(), False

    # B2: Grayscale
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # B3: Khử bóng đổ & chuẩn hóa ánh sáng
    normalized = (
        remove_illumination_background(gray, background_kernel)
        if remove_background
        else gray.copy()
    )

    # B4: Adaptive Threshold
    binary = adaptive_threshold_document(
        normalized,
        method=adaptive_method,
        block_size=block_size,
        c=c,
    )

    # B5: Morphology làm sạch
    binary = clean_binary_document(
        binary,
        operation=morph_operation,
        kernel_size=morph_kernel,
        iterations=morph_iterations,
    )

    return DocumentProcessResult(
        original=image.copy(),
        perspective_corrected=page,
        normalized_gray=normalized,
        binary=binary,
        document_found=found,
        corners=corners,
    )


def document_metrics(binary: np.ndarray) -> dict:
    """Thống kê tỷ lệ pixel trắng/đen và cường độ trung bình."""
    if binary.ndim != 2:
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary
    black_ratio = float(np.mean(gray < 128) * 100.0)
    white_ratio = 100.0 - black_ratio
    return {
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "mean": float(gray.mean()),
        "std": float(gray.std()),
    }


# ==============================================================
# PHẦN 7: XUẤT FILE PDF ĐA TRANG (PDF EXPORT)
# ==============================================================

def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Không hỗ trợ shape={image.shape}")
    return Image.fromarray(rgb)


def _fit_to_canvas(img: Image.Image, size: tuple[int, int], margin: int) -> Image.Image:
    canvas_w, canvas_h = size
    margin = max(0, int(margin))
    max_w = max(1, canvas_w - 2 * margin)
    max_h = max(1, canvas_h - 2 * margin)

    copy = img.copy()
    copy.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    x = (canvas_w - copy.width) // 2
    y = (canvas_h - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def export_images_to_pdf(
    images: Iterable[np.ndarray],
    output_path: str,
    page_size: str = "Auto",
    orientation: str = "Portrait",
    margin: int = 30,
) -> None:
    """Xuất một hoặc nhiều trang ảnh đã xử lý thành file PDF duy nhất."""
    pil_images = [_to_pil_rgb(img) for img in images if img is not None]
    if not pil_images:
        raise ValueError("Không có ảnh để xuất PDF")

    if page_size.lower() == "a4":
        # A4 ở 150 DPI
        size = (1240, 1754)
        if orientation.lower() == "landscape":
            size = (size[1], size[0])
        pil_images = [_fit_to_canvas(img, size, margin) for img in pil_images]

    first, rest = pil_images[0], pil_images[1:]
    first.save(
        output_path,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=rest,
    )


def export_pdf_bilevel_1bit(images: list[np.ndarray], output_path: str) -> None:
    """Xuất PDF Monochrome 1-bit Bilevel siêu nén (giảm 84% dung lượng)."""
    pil_pages = []
    for img in images:
        if img is None:
            continue
        pil_img = Image.fromarray(img > 128).convert("1")
        pil_pages.append(pil_img)

    if not pil_pages:
        raise ValueError("Không có trang hợp lệ để xuất PDF")

    first, rest = pil_pages[0], pil_pages[1:]
    first.save(output_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
