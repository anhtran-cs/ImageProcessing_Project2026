from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Cho phép chạy trực tiếp từ thư mục project1_miniphotoshop.
CUR_DIR = Path(__file__).resolve().parent
ROOT = CUR_DIR.parent
for p in (CUR_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import read_image, save_image, set_label_image, make_image_label
from histogram import HistogramCanvas
from xulyanh import (
    adjust_brightness,
    adjust_contrast,
    apply_blur,
    histogram_equalization_color,
    image_statistics,
)


class MiniPhotoshopWindow(QMainWindow):
    """PROJECT 1 - MINI PHOTOSHOP"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project 1 - Mini Photoshop")
        self.resize(1500, 900)

        # ============================================================
        # TRẠNG THÁI ẢNH
        # ============================================================
        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.history: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []

        # Dùng khi preview slider để không cộng dồn sai.
        self._slider_base: Optional[np.ndarray] = None

        self._build_ui()
        self._build_menu()

    # ================================================================
    # PHẦN 1 - XÂY DỰNG GUI
    # ================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("Mở ảnh")
        self.btn_save = QPushButton("Lưu kết quả")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        for b in [self.btn_open, self.btn_save, self.btn_reset, self.btn_undo, self.btn_redo]:
            top_bar.addWidget(b)
        top_bar.addStretch()
        root.addLayout(top_bar)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        # Khối ảnh trước/sau.
        image_area = QVBoxLayout()
        image_row = QHBoxLayout()
        self.lbl_original = make_image_label("ORIGINAL")
        self.lbl_processed = make_image_label("PROCESSED")
        image_row.addWidget(self.lbl_original, 1)
        image_row.addWidget(self.lbl_processed, 1)
        image_area.addLayout(image_row, 3)

        # Histogram.
        hist_controls = QHBoxLayout()
        self.cbo_hist_mode = QComboBox()
        self.cbo_hist_mode.addItems(["Gray", "RGB"])
        self.spin_bins = QSpinBox()
        self.spin_bins.setRange(2, 256)
        self.spin_bins.setValue(256)
        self.btn_equalize = QPushButton("Histogram Equalization (mở rộng)")
        hist_controls.addWidget(QLabel("Mode:"))
        hist_controls.addWidget(self.cbo_hist_mode)
        hist_controls.addWidget(QLabel("Bins:"))
        hist_controls.addWidget(self.spin_bins)
        hist_controls.addWidget(self.btn_equalize)
        hist_controls.addStretch()
        image_area.addLayout(hist_controls)

        self.hist_canvas = HistogramCanvas()
        image_area.addWidget(self.hist_canvas, 2)

        body.addLayout(image_area, 4)

        # Panel điều khiển bên phải.
        controls = QVBoxLayout()
        controls.addWidget(self._brightness_group())
        controls.addWidget(self._contrast_group())
        controls.addWidget(self._blur_group())

        stats_group = QGroupBox("Thống kê / Đánh giá")
        stats_layout = QVBoxLayout(stats_group)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        stats_layout.addWidget(self.txt_stats)
        controls.addWidget(stats_group, 1)
        body.addLayout(controls, 1)

        # Signals chung.
        self.btn_open.clicked.connect(self.open_image)
        self.btn_save.clicked.connect(self.save_current_image)
        self.btn_reset.clicked.connect(self.reset_image)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.cbo_hist_mode.currentTextChanged.connect(self.refresh_views)
        self.spin_bins.valueChanged.connect(self.refresh_views)
        self.btn_equalize.clicked.connect(self.apply_histogram_equalization)

    def _brightness_group(self):
        group = QGroupBox("Brightness")
        layout = QVBoxLayout(group)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.lbl_brightness_value = QLabel("β = 0")
        layout.addWidget(self.lbl_brightness_value)
        layout.addWidget(self.slider_brightness)

        self.slider_brightness.sliderPressed.connect(self._begin_slider_preview)
        self.slider_brightness.valueChanged.connect(self.preview_brightness)
        self.slider_brightness.sliderReleased.connect(self.commit_brightness)
        return group

    def _contrast_group(self):
        group = QGroupBox("Contrast")
        layout = QVBoxLayout(group)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(10, 300)  # 0.10 -> 3.00
        self.slider_contrast.setValue(100)
        self.lbl_contrast_value = QLabel("α = 1.00")
        layout.addWidget(self.lbl_contrast_value)
        layout.addWidget(self.slider_contrast)

        self.slider_contrast.sliderPressed.connect(self._begin_slider_preview)
        self.slider_contrast.valueChanged.connect(self.preview_contrast)
        self.slider_contrast.sliderReleased.connect(self.commit_contrast)
        return group

    def _blur_group(self):
        group = QGroupBox("Blur")
        form = QFormLayout(group)
        self.cbo_blur = QComboBox()
        self.cbo_blur.addItems(["Gaussian", "Mean", "Median"])
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 31)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(5)
        self.btn_blur = QPushButton("Áp dụng Blur")
        form.addRow("Method:", self.cbo_blur)
        form.addRow("Kernel:", self.spin_kernel)
        form.addRow(self.btn_blur)
        self.btn_blur.clicked.connect(self.apply_blur_from_ui)
        return group

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        act_open = QAction("Open", self)
        act_save = QAction("Save As", self)
        act_exit = QAction("Exit", self)
        file_menu.addActions([act_open, act_save, act_exit])
        act_open.triggered.connect(self.open_image)
        act_save.triggered.connect(self.save_current_image)
        act_exit.triggered.connect(self.close)

    # ================================================================
    # PHẦN 2 - ĐỌC / LƯU ẢNH
    # ================================================================
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            image = read_image(path)
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.preview_image = None
            self.history.clear()
            self.redo_stack.clear()
            self._reset_controls_without_processing()
            self.refresh_views()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", str(exc))

    def save_current_image(self):
        image = self._display_image()
        if image is None:
            self._warn_no_image()
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh",
            "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        try:
            save_image(path, image)
            QMessageBox.information(self, "Thành công", "Đã lưu ảnh.")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", str(exc))

    # ================================================================
    # PHẦN 3 - BRIGHTNESS / CONTRAST REAL-TIME PREVIEW
    # ================================================================
    def _begin_slider_preview(self):
        if self.current_image is not None:
            self._slider_base = self.current_image.copy()

    def preview_brightness(self, value: int):
        self.lbl_brightness_value.setText(f"β = {value}")
        if self.current_image is None:
            return
        base = self._slider_base if self._slider_base is not None else self.current_image
        self.preview_image = adjust_brightness(base, value)
        self.refresh_views()

    def commit_brightness(self):
        if self.current_image is None or self._slider_base is None:
            return
        value = self.slider_brightness.value()
        if value != 0:
            self._push_history(self._slider_base)
            self.current_image = adjust_brightness(self._slider_base, value)
        self.preview_image = None
        self._slider_base = None
        self.slider_brightness.blockSignals(True)
        self.slider_brightness.setValue(0)
        self.slider_brightness.blockSignals(False)
        self.lbl_brightness_value.setText("β = 0")
        self.refresh_views()

    def preview_contrast(self, raw_value: int):
        alpha = raw_value / 100.0
        self.lbl_contrast_value.setText(f"α = {alpha:.2f}")
        if self.current_image is None:
            return
        base = self._slider_base if self._slider_base is not None else self.current_image
        self.preview_image = adjust_contrast(base, alpha)
        self.refresh_views()

    def commit_contrast(self):
        if self.current_image is None or self._slider_base is None:
            return
        alpha = self.slider_contrast.value() / 100.0
        if abs(alpha - 1.0) > 1e-9:
            self._push_history(self._slider_base)
            self.current_image = adjust_contrast(self._slider_base, alpha)
        self.preview_image = None
        self._slider_base = None
        self.slider_contrast.blockSignals(True)
        self.slider_contrast.setValue(100)
        self.slider_contrast.blockSignals(False)
        self.lbl_contrast_value.setText("α = 1.00")
        self.refresh_views()

    # ================================================================
    # PHẦN 4 - BLUR
    # ================================================================
    def apply_blur_from_ui(self):
        if self.current_image is None:
            self._warn_no_image()
            return
        k = self.spin_kernel.value()
        if k % 2 == 0:
            QMessageBox.warning(self, "Kernel không hợp lệ", "Kernel phải là số lẻ.")
            return
        try:
            self._push_history(self.current_image)
            self.current_image = apply_blur(
                self.current_image,
                self.cbo_blur.currentText(),
                k,
            )
            self.preview_image = None
            self.refresh_views()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", str(exc))

    # ================================================================
    # PHẦN 5 - HISTOGRAM / HISTOGRAM EQUALIZATION
    # ================================================================
    def apply_histogram_equalization(self):
        if self.current_image is None:
            self._warn_no_image()
            return
        self._push_history(self.current_image)
        self.current_image = histogram_equalization_color(self.current_image)
        self.refresh_views()

    # ================================================================
    # PHẦN 6 - RESET / UNDO / REDO
    # ================================================================
    def _push_history(self, image: np.ndarray):
        self.history.append(image.copy())
        if len(self.history) > 30:
            self.history.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.current_image is None or not self.history:
            return
        self.redo_stack.append(self.current_image.copy())
        self.current_image = self.history.pop()
        self.preview_image = None
        self.refresh_views()

    def redo(self):
        if self.current_image is None or not self.redo_stack:
            return
        self.history.append(self.current_image.copy())
        self.current_image = self.redo_stack.pop()
        self.preview_image = None
        self.refresh_views()

    def reset_image(self):
        if self.original_image is None:
            return
        if self.current_image is not None:
            self._push_history(self.current_image)
        self.current_image = self.original_image.copy()
        self.preview_image = None
        self._reset_controls_without_processing()
        self.refresh_views()

    def _reset_controls_without_processing(self):
        self.slider_brightness.blockSignals(True)
        self.slider_contrast.blockSignals(True)
        self.slider_brightness.setValue(0)
        self.slider_contrast.setValue(100)
        self.slider_brightness.blockSignals(False)
        self.slider_contrast.blockSignals(False)
        self.lbl_brightness_value.setText("β = 0")
        self.lbl_contrast_value.setText("α = 1.00")

    # ================================================================
    # PHẦN 7 - HIỂN THỊ & ĐÁNH GIÁ
    # ================================================================
    def _display_image(self):
        return self.preview_image if self.preview_image is not None else self.current_image

    def refresh_views(self, *_args):
        set_label_image(self.lbl_original, self.original_image)
        shown = self._display_image()
        set_label_image(self.lbl_processed, shown)

        try:
            self.hist_canvas.update_histogram(
                shown,
                mode=self.cbo_hist_mode.currentText(),
                bins=self.spin_bins.value(),
            )
        except Exception:
            pass

        if shown is not None:
            s = image_statistics(shown)
            text = (
                f"Shape: {s['shape']}\n"
                f"dtype: {s['dtype']}\n"
                f"Min: {s['min']}\n"
                f"Max: {s['max']}\n"
                f"Mean: {s['mean']:.3f}\n"
                f"Std: {s['std']:.3f}\n"
                f"Dynamic range: {s['dynamic_range']}"
            )
            self.txt_stats.setPlainText(text)
        else:
            self.txt_stats.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_views()

    def _warn_no_image(self):
        QMessageBox.warning(self, "Chưa có ảnh", "Hãy mở một ảnh trước.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MiniPhotoshopWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
