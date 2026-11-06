"""Smoke test thuật toán lõi. Không mở GUI."""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mini_photoshop"))
sys.path.insert(0, str(ROOT / "document_scanner"))
sys.path.insert(0, str(ROOT))

from filters import adjust_brightness, adjust_contrast, apply_blur, compute_histogram
from doc_processor import process_document, export_images_to_pdf


def main():
    # Ảnh màu tổng hợp cho Project 1.
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :160] = (30, 80, 120)
    img[:, 160:] = (220, 180, 120)

    assert adjust_brightness(img, 50).dtype == np.uint8
    assert adjust_contrast(img, 1.5).dtype == np.uint8
    assert apply_blur(img, "Gaussian", 5).shape == img.shape
    hist = compute_histogram(img, "Gray", 256)
    assert "Gray" in hist

    # Tài liệu tổng hợp cho Project 2.
    doc = np.full((600, 450, 3), 230, dtype=np.uint8)
    cv2.rectangle(doc, (25, 25), (425, 575), (255, 255, 255), -1)
    for y in range(80, 540, 45):
        cv2.putText(doc, "Document test line", (55, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (20, 20, 20), 2, cv2.LINE_AA)

    result = process_document(
        doc,
        detect_page=True,
        remove_background=True,
        background_kernel=31,
        adaptive_method="Gaussian",
        block_size=21,
        c=10,
        morph_operation="None",
    )
    assert result.binary.dtype == np.uint8
    assert result.binary.ndim == 2

    pdf_path = ROOT / "tests" / "_smoke_output.pdf"
    export_images_to_pdf([result.binary, result.binary], str(pdf_path), page_size="A4")
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    pdf_path.unlink(missing_ok=True)
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
