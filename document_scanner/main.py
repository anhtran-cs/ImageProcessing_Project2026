from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

CUR_DIR = Path(__file__).resolve().parent
ROOT = CUR_DIR.parent
for p in (CUR_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import read_image, save_image, set_label_image, make_image_label
from xulytailieu import (
    document_metrics,
    process_document,
    export_images_to_pdf,
    draw_document_corners,
    detect_document_corners,
)


@dataclass
class PageData:
    path: str
    original: np.ndarray
    processed: Optional[np.ndarray] = None
    document_found: bool = False
    corners: Optional[np.ndarray] = None


class DocumentDigitizationWindow(QMainWindow):
    """PROJECT 2 - ỨNG DỤNG SỐ HÓA TÀI LIỆU"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project 2 - Ứng dụng số hóa tài liệu - N10")
        self.resize(1550, 900)
        self.pages: list[PageData] = []
        self._build_ui()
        
        #GUI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        self.btn_add = QPushButton("Thêm ảnh")
        self.btn_remove = QPushButton("Xóa trang")
        self.btn_up = QPushButton("↑ Lên")
        self.btn_down = QPushButton("↓ Xuống")
        self.list_pages = QListWidget()

        row_add = QHBoxLayout()
        row_add.addWidget(self.btn_add)
        row_add.addWidget(self.btn_remove)
        left.addLayout(row_add)
        left.addWidget(QLabel("Danh sách trang:"))
        left.addWidget(self.list_pages, 1)
        row_order = QHBoxLayout()
        row_order.addWidget(self.btn_up)
        row_order.addWidget(self.btn_down)
        left.addLayout(row_order)
        root.addLayout(left, 1)

        center = QVBoxLayout()
        header_row = QHBoxLayout()
        self.chk_show_corners = QCheckBox("Hiện khung quét 4 góc trên ảnh gốc")
        self.chk_show_corners.setChecked(True)
        self.chk_show_corners.setStyleSheet("font-weight: bold; color: #48bb78;")
        header_row.addWidget(self.chk_show_corners)
        header_row.addStretch()
        center.addLayout(header_row)

        image_row = QHBoxLayout()
        self.lbl_original = make_image_label("ORIGINAL DOCUMENT")
        self.lbl_processed = make_image_label("PROCESSED DOCUMENT")
        image_row.addWidget(self.lbl_original, 1)
        image_row.addWidget(self.lbl_processed, 1)
        center.addLayout(image_row, 4)

        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setMaximumHeight(150)
        center.addWidget(self.txt_metrics)
        root.addLayout(center, 4)

        controls = QVBoxLayout()
        controls.addWidget(self._preprocess_group())
        controls.addWidget(self._threshold_group())
        controls.addWidget(self._morphology_group())
        controls.addWidget(self._export_group())

        self.btn_process_current = QPushButton("Xử lý trang hiện tại")
        self.btn_process_all = QPushButton("Xử lý TẤT CẢ")
        self.btn_save_current = QPushButton("Lưu ảnh hiện tại")
        controls.addWidget(self.btn_process_current)
        controls.addWidget(self.btn_process_all)
        controls.addWidget(self.btn_save_current)
        controls.addStretch()
        root.addLayout(controls, 1)

        self.btn_add.clicked.connect(self.add_images)
        self.btn_remove.clicked.connect(self.remove_current_page)
        self.btn_up.clicked.connect(lambda: self.move_page(-1))
        self.btn_down.clicked.connect(lambda: self.move_page(1))
        self.list_pages.currentRowChanged.connect(self.refresh_preview)
        self.chk_show_corners.toggled.connect(self.refresh_preview)
        self.spin_inset.valueChanged.connect(self.refresh_preview)
        self.btn_process_current.clicked.connect(self.process_current)
        self.btn_process_all.clicked.connect(self.process_all)
        self.btn_save_current.clicked.connect(self.save_current)
        self.btn_export_pdf.clicked.connect(self.export_pdf)

    def _preprocess_group(self):
        group = QGroupBox("1. Tiền xử lý / Loại bỏ nền")
        form = QFormLayout(group)
        self.chk_detect_page = QCheckBox("Tách giấy khỏi nền + perspective")
        self.chk_detect_page.setChecked(True)
        self.spin_inset = QDoubleSpinBox()
        self.spin_inset.setRange(0.0, 5.0)
        self.spin_inset.setSingleStep(0.5)
        self.spin_inset.setValue(1.5)
        self.spin_inset.setSuffix("%")
        self.chk_remove_bg = QCheckBox("Khử nền/ánh sáng không đều")
        self.chk_remove_bg.setChecked(True)
        self.spin_bg_kernel = QSpinBox()
        self.spin_bg_kernel.setRange(3, 151)
        self.spin_bg_kernel.setSingleStep(2)
        self.spin_bg_kernel.setValue(51)
        form.addRow(self.chk_detect_page)
        form.addRow("Co lề cắt mép (Inset):", self.spin_inset)
        form.addRow(self.chk_remove_bg)
        form.addRow("Background kernel:", self.spin_bg_kernel)
        return group

    def _threshold_group(self):
        group = QGroupBox("2. Adaptive Threshold")
        form = QFormLayout(group)
        self.cbo_adaptive = QComboBox()
        self.cbo_adaptive.addItems(["Gaussian", "Mean"])
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 101)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(21)
        self.spin_c = QDoubleSpinBox()
        self.spin_c.setRange(-30.0, 30.0)
        self.spin_c.setDecimals(1)
        self.spin_c.setValue(10.0)
        form.addRow("Method:", self.cbo_adaptive)
        form.addRow("Block size:", self.spin_block)
        form.addRow("C:", self.spin_c)
        return group

    def _morphology_group(self):
        group = QGroupBox("3. Morphology làm sạch")
        form = QFormLayout(group)
        self.cbo_morph = QComboBox()
        self.cbo_morph.addItems(["None", "Opening", "Closing"])
        self.cbo_morph.setCurrentText("None")
        self.spin_morph_kernel = QSpinBox()
        self.spin_morph_kernel.setRange(1, 9)
        self.spin_morph_kernel.setValue(3)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(1, 5)
        self.spin_iterations.setValue(1)
        form.addRow("Operation:", self.cbo_morph)
        form.addRow("Kernel:", self.spin_morph_kernel)
        form.addRow("Iterations:", self.spin_iterations)
        return group

    def _export_group(self):
        group = QGroupBox("4. Xuất PDF")
        form = QFormLayout(group)
        self.cbo_page_size = QComboBox()
        self.cbo_page_size.addItems(["Auto", "A4"])
        self.cbo_orientation = QComboBox()
        self.cbo_orientation.addItems(["Portrait", "Landscape"])
        self.spin_margin = QSpinBox()
        self.spin_margin.setRange(0, 200)
        self.spin_margin.setValue(30)
        self.btn_export_pdf = QPushButton("Export PDF")
        form.addRow("Page size:", self.cbo_page_size)
        form.addRow("Orientation:", self.cbo_orientation)
        form.addRow("Margin:", self.spin_margin)
        form.addRow(self.btn_export_pdf)
        return group

    # ================================================================
    # PHẦN 2 - QUẢN LÝ ẢNH / TRANG
    # ================================================================
    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh tài liệu",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not paths:
            return

        errors = []
        for path in paths:
            try:
                image = read_image(path)
                self.pages.append(PageData(path=path, original=image))
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        self.rebuild_page_list()
        if self.pages and self.list_pages.currentRow() < 0:
            self.list_pages.setCurrentRow(0)
        if errors:
            QMessageBox.warning(self, "Một số file lỗi", "\n".join(errors))

    def rebuild_page_list(self):
        current = self.list_pages.currentRow()
        self.list_pages.clear()
        for i, page in enumerate(self.pages, start=1):
            state = "✓" if page.processed is not None else "•"
            item = QListWidgetItem(f"{state} {i:03d}. {Path(page.path).name}")
            self.list_pages.addItem(item)
        if self.pages:
            self.list_pages.setCurrentRow(min(max(current, 0), len(self.pages) - 1))

    def remove_current_page(self):
        row = self.list_pages.currentRow()
        if 0 <= row < len(self.pages):
            self.pages.pop(row)
            self.rebuild_page_list()
            self.refresh_preview()

    def move_page(self, delta: int):
        row = self.list_pages.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= len(self.pages):
            return
        self.pages[row], self.pages[target] = self.pages[target], self.pages[row]
        self.rebuild_page_list()
        self.list_pages.setCurrentRow(target)

    # ================================================================
    # PHẦN 3 - PIPELINE XỬ LÝ TÀI LIỆU
    # ================================================================
    def _params(self) -> dict:
        block = self.spin_block.value()
        bg_k = self.spin_bg_kernel.value()
        if block % 2 == 0 or block <= 1:
            raise ValueError("Block size phải là số lẻ > 1")
        if bg_k % 2 == 0:
            raise ValueError("Background kernel phải là số lẻ")
        return {
            "detect_page": self.chk_detect_page.isChecked(),
            "inset_ratio": self.spin_inset.value() / 100.0,
            "remove_background": self.chk_remove_bg.isChecked(),
            "background_kernel": bg_k,
            "adaptive_method": self.cbo_adaptive.currentText(),
            "block_size": block,
            "c": self.spin_c.value(),
            "morph_operation": self.cbo_morph.currentText(),
            "morph_kernel": self.spin_morph_kernel.value(),
            "morph_iterations": self.spin_iterations.value(),
        }

    def process_current(self):
        row = self.list_pages.currentRow()
        if row < 0 or row >= len(self.pages):
            self._warn_no_page()
            return
        try:
            result = process_document(self.pages[row].original, **self._params())
            self.pages[row].processed = result.binary
            self.pages[row].document_found = result.document_found
            self.pages[row].corners = result.corners
            self.rebuild_page_list()
            self.list_pages.setCurrentRow(row)
            self.refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xử lý", str(exc))

    def process_all(self):
        if not self.pages:
            self._warn_no_page()
            return
        try:
            params = self._params()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            for page in self.pages:
                result = process_document(page.original, **params)
                page.processed = result.binary
                page.document_found = result.document_found
                page.corners = result.corners
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xử lý", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
        self.rebuild_page_list()
        self.refresh_preview()

    # ================================================================
    # PHẦN 4 - SAVE ẢNH / EXPORT PDF
    # ================================================================
    def save_current(self):
        row = self.list_pages.currentRow()
        if row < 0 or row >= len(self.pages):
            self._warn_no_page()
            return
        image = self.pages[row].processed
        if image is None:
            QMessageBox.warning(self, "Chưa xử lý", "Hãy xử lý trang trước khi lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh tài liệu",
            "document_page.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        try:
            save_image(path, image)
            QMessageBox.information(self, "Thành công", "Đã lưu ảnh.")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", str(exc))

    def export_pdf(self):
        if not self.pages:
            self._warn_no_page()
            return

        missing = [i + 1 for i, p in enumerate(self.pages) if p.processed is None]
        if missing:
            answer = QMessageBox.question(
                self,
                "Có trang chưa xử lý",
                f"Các trang {missing} chưa được xử lý. Xử lý tất cả trước khi export?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self.process_all()
            else:
                return

        images = [p.processed for p in self.pages if p.processed is not None]
        if not images:
            QMessageBox.warning(self, "Không có dữ liệu", "Không có trang đã xử lý.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất PDF",
            "document.pdf",
            "PDF (*.pdf)",
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        try:
            export_images_to_pdf(
                images,
                path,
                page_size=self.cbo_page_size.currentText(),
                orientation=self.cbo_orientation.currentText(),
                margin=self.spin_margin.value(),
            )
            QMessageBox.information(self, "Thành công", f"Đã xuất PDF:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xuất PDF", str(exc))

    # ================================================================
    # PHẦN 5 - HIỂN THỊ / ĐÁNH GIÁ
    # ================================================================
    def refresh_preview(self, *_args):
        row = self.list_pages.currentRow()
        if row < 0 or row >= len(self.pages):
            set_label_image(self.lbl_original, None)
            set_label_image(self.lbl_processed, None)
            self.txt_metrics.clear()
            return

        page = self.pages[row]
        # Hiển thị ảnh gốc: có hoặc không có khung quét 4 góc
        if self.chk_show_corners.isChecked():
            if page.corners is None:
                c, _ = detect_document_corners(page.original)
                page.corners = c
                page.document_found = c is not None
            inset = self.spin_inset.value() / 100.0 if hasattr(self, "spin_inset") else 0.015
            display_orig = draw_document_corners(page.original, page.corners, inset)
            set_label_image(self.lbl_original, display_orig)
        else:
            set_label_image(self.lbl_original, page.original)

        set_label_image(self.lbl_processed, page.processed)

        if page.processed is None:
            self.txt_metrics.setPlainText("Trang chưa được xử lý.")
        else:
            m = document_metrics(page.processed)
            detected = "Có" if page.document_found else "Không / dùng ảnh gốc"
            self.txt_metrics.setPlainText(
                f"Tìm thấy biên tài liệu: {detected}\n"
                f"Black pixel ratio: {m['black_ratio']:.2f}%\n"
                f"White pixel ratio: {m['white_ratio']:.2f}%\n"
                f"Mean: {m['mean']:.3f}\n"
                f"Std: {m['std']:.3f}"
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_preview()

    def _warn_no_page(self):
        QMessageBox.warning(self, "Chưa có trang", "Hãy thêm ảnh tài liệu trước.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DocumentDigitizationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
