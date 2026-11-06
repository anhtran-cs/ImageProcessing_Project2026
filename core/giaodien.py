from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy


def make_image_label(title: str) -> QLabel:
    label = QLabel(title)
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(420, 320)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.setStyleSheet(
        "QLabel { background: #1f2329; color: #c9d1d9; "
        "border: 1px solid #4b5563; border-radius: 6px; }"
    )
    return label
