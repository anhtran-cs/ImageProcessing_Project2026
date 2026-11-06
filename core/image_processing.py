from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def read_image(path: str) -> np.ndarray:
    """
    Đọc ảnh màu theo chuẩn OpenCV:
    - Output: BGR
    - dtype: uint8
    - Hỗ trợ tên file / đường dẫn Unicode trên Windows
    """

    path_obj = Path(path)

    # Kiểm tra file có tồn tại không
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh:\n{path}")

    try:
        # Đọc file thành mảng byte
        data = np.fromfile(str(path_obj), dtype=np.uint8)

        # Giải mã thành ảnh OpenCV
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    except Exception as exc:
        raise ValueError(
            f"Lỗi khi đọc ảnh:\n{path}\n\nChi tiết: {exc}"
        )

    if image is None:
        raise ValueError(
            f"Không thể giải mã ảnh:\n{path}\n\n"
            "Hãy kiểm tra định dạng hoặc thử ảnh JPG/PNG khác."
        )

    return image


def save_image(path: str, image: np.ndarray) -> None:
    """
    Lưu ảnh và hỗ trợ đường dẫn Unicode trên Windows.
    """

    if image is None or image.size == 0:
        raise ValueError("Ảnh rỗng, không thể lưu.")

    path_obj = Path(path)

    # Nếu người dùng không nhập extension
    extension = path_obj.suffix.lower()

    if extension == "":
        extension = ".png"
        path_obj = path_obj.with_suffix(extension)

    if extension not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        raise ValueError(
            f"Định dạng không hỗ trợ: {extension}"
        )

    # JPEG dùng .jpg khi encode
    encode_ext = ".jpg" if extension == ".jpeg" else extension

    success, encoded = cv2.imencode(
        encode_ext,
        image
    )

    if not success:
        raise IOError(
            f"Không thể mã hóa ảnh để lưu:\n{path_obj}"
        )

    encoded.tofile(str(path_obj))


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Đưa ảnh về uint8 an toàn sau các phép toán trung gian."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def cv_to_qimage(image: np.ndarray) -> QImage:
    """Chuyển ndarray OpenCV sang QImage để hiển thị trong PySide6."""
    if image is None:
        return QImage()

    image = ensure_uint8(image)

    if image.ndim == 2:
        h, w = image.shape
        return QImage(
            image.data, w, h, image.strides[0], QImage.Format_Grayscale8
        ).copy()

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        return QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
        ).copy()

    if image.ndim == 3 and image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        h, w, _ = rgba.shape
        return QImage(
            rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888
        ).copy()

    raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={image.shape}")


def set_label_image(label: QLabel, image: Optional[np.ndarray]) -> None:
    """Hiển thị ảnh vào QLabel và tự co theo kích thước khung."""
    if image is None:
        label.clear()
        label.setText("Chưa có ảnh")
        return

    pixmap = QPixmap.fromImage(cv_to_qimage(image))
    if label.width() > 10 and label.height() > 10:
        pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)


def image_name(path: str) -> str:
    return Path(path).name
