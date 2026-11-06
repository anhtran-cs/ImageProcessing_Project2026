# Computer Vision Studio & Document Scanner

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt-brightgreen.svg)](https://doc.qt.io/qtforpython/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-red.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end desktop computer vision application suite built with **Python**, **PySide6 (Qt)**, **OpenCV**, and **NumPy**. The project comprises two production-ready desktop tools:

1. **Mini Photoshop**: An interactive digital image manipulation studio featuring spatial filtering, brightness/contrast scaling, and dynamic multi-channel histogram equalization.
2. **Smart Document Scanner**: A document digitization pipeline with automated 4-point corner detection, perspective transformation, illumination correction, adaptive thresholding, morphological refinement, and multi-page PDF compilation.

---

## Architecture Overview

```text
ChuDe3_XuLyAnh/
├── core/                           # Shared utility modules
│   ├── __init__.py                 # Exported core interfaces
│   ├── gui.py                      # Common UI components & themes
│   └── image_processing.py         # Unicode-safe image I/O & Qt format conversions
├── mini_photoshop/                 # Application 1: Digital Image Editor
│   ├── __init__.py
│   ├── filters.py                  # Core image processing operators & filters
│   ├── histogram.py                # Real-time Matplotlib histogram canvas
│   └── main.py                     # Mini Photoshop PySide6 GUI
├── document_scanner/               # Application 2: Document Scanner & Digitizer
│   ├── __init__.py
│   ├── doc_processor.py            # Perspective warp, thresholding, PDF export engine
│   └── main.py                     # Document Scanner PySide6 GUI
├── tests/                          # Automated test suite
│   ├── test_smoke.py               # Fast headless algorithm smoke tests (<1s)
│   └── test_comprehensive.py       # Full unit & stress test suite (100+ assertions)
├── .gitignore                      # Git exclusion rules (pycache, virtualenvs, etc.)
├── LICENSE                         # MIT License
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## Application 1: Mini Photoshop

A desktop image editor providing essential point operations, neighborhood spatial filtering, and photometric analysis.

### Core Features
- **Format Support**: Multi-format image import and export (PNG, JPG/JPEG, BMP, TIFF) with full Unicode path compatibility on Windows.
- **Intensity Adjustment**:
  - **Brightness**: Linear intensity shift $g(x,y) = \text{clip}(f(x,y) + \beta, 0, 255)$.
  - **Contrast**: Symmetric scaling around midpoint $g(x,y) = \text{clip}(\alpha \cdot (f(x,y) - \text{pivot}) + \text{pivot}, 0, 255)$.
- **Spatial Convolution Filtering**:
  - **Gaussian Blur**: Kernel-based weighted spatial smoothing.
  - **Mean Box Blur**: Neighborhood averaging.
  - **Median Blur**: Non-linear filtering for impulse (salt-and-pepper) noise reduction.
- **Histogram Equalization**: Multi-channel dynamic range expansion in HSV/YCrCb color spaces to preserve natural chroma while maximizing contrast.
- **Real-Time Histogram Visualization**: Dynamic single-channel (Grayscale) and multi-channel (RGB) histogram plotting via embedded Matplotlib canvas.
- **Productivity Controls**: Infinite Undo/Redo stack, side-by-side Before/After preview, and statistical metric reporting (Mean, Std Dev, Min/Max intensity).

---

## Application 2: Smart Document Scanner

An automated document extraction engine designed to convert distorted or poorly-lit camera photos into clean, high-contrast, archivable scanned documents.

### Processing Pipeline

```
Raw Camera Input
       │
       ▼
[1. Document Boundary Detection]
  - Morphological edge filtering, contour hierarchy & convex hull
  - 4-point corner sorting (TL, TR, BR, BL) using polar-angle ordering
       │
       ▼
[2. Homography & 4-Point Perspective Transform]
  - Homography matrix calculation via cv2.getPerspectiveTransform
  - Warping and aspect-ratio normalization to target A4 dimensions
       │
       ▼
[3. Non-Uniform Illumination Removal]
  - Morphological background approximation (large structural element)
  - Division / subtraction normalization to eliminate shadows and gradients
       │
       ▼
[4. Adaptive Binarization]
  - Gaussian-weighted adaptive thresholding (Otsu & Sauvola compatible)
  - Local threshold tuning based on neighborhood pixel distributions
       │
       ▼
[5. Morphological Post-Processing]
  - Noise suppression and stroke enhancement via morphological open/close
       │
       ▼
[6. Output Compilation]
  - High-resolution TIFF/PNG export and multi-page A4 PDF packaging
```

---

## Getting Started

### Prerequisites
- Python **3.10** or higher
- Windows, macOS, or Linux

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tranhoanganhuser-wq/xla26.git
   cd xla26
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run Mini Photoshop
```bash
python mini_photoshop/main.py
```

### Run Document Scanner
```bash
python document_scanner/main.py
```

---

## Testing & Quality Assurance

The repository includes headless test suites verifying algorithmic correctness, edge cases, and performance without opening GUI windows.

```bash
# Run quick smoke test (< 1 second)
python tests/test_smoke.py

# Run comprehensive algorithm suite (100+ assertions across 10 test groups)
python tests/test_comprehensive.py
```

---

## Key Technical Conventions

- **Mathematical Pipeline**: All intermediate numerical calculations convert `uint8 -> float32 -> transform -> np.clip(0, 255) -> uint8` to avoid 8-bit integer overflow/underflow artifacts.
- **Decoupled Architecture**: UI elements in `main.py` act strictly as presentation controllers; core vision algorithms reside independently in functional, testable utility modules.
- **Cross-Platform Compatibility**: File paths utilize `pathlib.Path` with byte-stream decoders (`np.fromfile` + `cv2.imdecode`) ensuring robust handling of UTF-8 and Unicode file paths on Windows.

---

## Author

- **Hoang Anh Tran** ([@tranhoanganhuser-wq](https://github.com/tranhoanganhuser-wq))
- Focus: Computer Science, Computer Vision, Data Science & Machine Learning.
