from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def adjust_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """
    PHẦN BRIGHTNESS
    g(x,y) = f(x,y) + beta
    Pipeline an toàn: uint8 -> float32 -> transform -> clip -> uint8.
    """
    x = image.astype(np.float32)
    y = x + float(beta)
    return np.clip(y, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float, pivot: float = 127.5) -> np.ndarray:
    """
    PHẦN CONTRAST
    Tăng/giảm độ tương phản quanh mức pivot:
        g = alpha * (f - pivot) + pivot
    alpha < 1: giảm contrast; alpha = 1: giữ nguyên; alpha > 1: tăng contrast.
    """
    if alpha <= 0:
        raise ValueError("alpha phải > 0")
    x = image.astype(np.float32)
    y = float(alpha) * (x - float(pivot)) + float(pivot)
    return np.clip(y, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, method: str, kernel_size: int) -> np.ndarray:
    """PHẦN BLUR: Mean / Gaussian / Median."""
    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError("kernel_size phải là số lẻ dương")

    method = method.lower().strip()
    if method == "gaussian":
        return cv2.GaussianBlur(image, (k, k), 0)
    if method == "mean":
        return cv2.blur(image, (k, k))
    if method == "median":
        return cv2.medianBlur(image, k)
    raise ValueError(f"Blur method không hỗ trợ: {method}")


def histogram_equalization_color(image: np.ndarray) -> np.ndarray:
    """
    PHẦN HISTOGRAM EQUALIZATION (mở rộng):
    cân bằng histogram trên kênh Y để hạn chế biến đổi màu.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def compute_histogram(
    image: np.ndarray,
    mode: str = "Gray",
    bins: int = 256,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    PHẦN HISTOGRAM
    Trả về dict {tên_kênh: (x, hist)}.
    bins có thể thay đổi để phục vụ yêu cầu điều chỉnh tham số.
    """
    bins = int(bins)
    if bins < 2 or bins > 256:
        raise ValueError("bins phải nằm trong [2, 256]")

    mode = mode.lower().strip()
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if image.ndim == 2 or mode == "gray":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
        x = np.linspace(0, 255, bins)
        result["Gray"] = (x, hist)
        return result

    if mode == "rgb":
        # Ảnh OpenCV là BGR nhưng tên hiển thị theo màu thực.
        for channel_index, name in [(2, "Red"), (1, "Green"), (0, "Blue")]:
            hist = cv2.calcHist([image], [channel_index], None, [bins], [0, 256]).ravel()
            x = np.linspace(0, 255, bins)
            result[name] = (x, hist)
        return result

    raise ValueError("mode phải là 'Gray' hoặc 'RGB'")


def image_statistics(image: np.ndarray) -> dict:
    """PHẦN ĐÁNH GIÁ: min/max/mean/std/dynamic range."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mn = int(gray.min())
    mx = int(gray.max())
    return {
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "min": mn,
        "max": mx,
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dynamic_range": int(mx - mn),
    }


def clipping_ratio_before_clip(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> tuple[float, float]:
    """Tỷ lệ % pixel dự kiến <0 và >255 trước clip cho biến đổi alpha*f+beta."""
    x = image.astype(np.float32)
    y = float(alpha) * x + float(beta)
    below = float(np.mean(y < 0) * 100.0)
    above = float(np.mean(y > 255) * 100.0)
    return below, above
