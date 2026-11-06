from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from xulyanh import compute_histogram


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 2.4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Histogram")
        self.ax.set_xlim(0, 255)

    def update_histogram(self, image, mode="Gray", bins=256):
        self.ax.clear()
        self.ax.set_title(f"Histogram - {mode} - {bins} bins")
        self.ax.set_xlabel("Cường độ")
        self.ax.set_ylabel("Tần số")
        self.ax.set_xlim(0, 255)

        if image is None:
            self.draw_idle()
            return

        data = compute_histogram(image, mode=mode, bins=bins)
        for name, (x, hist) in data.items():
            # Không cố định màu để code đơn giản; label đủ phân biệt.
            self.ax.plot(x, hist, label=name)
        if len(data) > 1:
            self.ax.legend()
        self.draw_idle()
