#!/usr/bin/env python3
import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QFileDialog, QLabel, QGroupBox, QStatusBar,
    QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from tomosar.core import SceneStats

class TomogramPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tomogram Dual Y-Axis Plotter")
        self.scene = None
        self.left_data = []
        self.right_data = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Top controls
        selector_group = QGroupBox("Tomogram Selection")
        selector_layout = QHBoxLayout()

        self.load_button = QPushButton("Load .tomo Folder")
        self.load_button.clicked.connect(self.load_tomo_folder)

        self.band_selector = QComboBox()
        self.mask_selector = QComboBox()
        self.layer_selector = QComboBox()
        self.layer_selector.addItems(["raw", "multilooked", "filtered"])

        selector_layout.addWidget(self.load_button)
        selector_layout.addWidget(QLabel("Band:"))
        selector_layout.addWidget(self.band_selector)
        selector_layout.addWidget(QLabel("Mask:"))
        selector_layout.addWidget(self.mask_selector)
        selector_layout.addWidget(QLabel("Layer:"))
        selector_layout.addWidget(self.layer_selector)

        selector_group.setLayout(selector_layout)
        self.main_layout.addWidget(selector_group)

        # Middle layout: left list, canvas, right list
        middle_layout = QHBoxLayout()

        # Left panel
        left_panel = QVBoxLayout()
        self.left_y_selector = QComboBox()
        self.add_left_button = QPushButton("Add to Left Axis")
        self.left_list = QListWidget()
        left_panel.addWidget(QLabel("Left Y Variable:"))
        left_panel.addWidget(self.left_y_selector)
        left_panel.addWidget(self.add_left_button)
        left_panel.addWidget(QLabel("Left Axis Tomograms:"))
        left_panel.addWidget(self.left_list)
        middle_layout.addLayout(left_panel)

        # Canvas and toolbar
        canvas_panel = QVBoxLayout()
        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_panel.addWidget(self.toolbar)
        canvas_panel.addWidget(self.canvas)
        middle_layout.addLayout(canvas_panel, stretch=1)

        # Right panel
        right_panel = QVBoxLayout()
        self.right_y_selector = QComboBox()
        self.add_right_button = QPushButton("Add to Right Axis")
        self.right_list = QListWidget()
        right_panel.addWidget(QLabel("Right Y Variable:"))
        right_panel.addWidget(self.right_y_selector)
        right_panel.addWidget(self.add_right_button)
        right_panel.addWidget(QLabel("Loaded statistics:"))
        right_panel.addWidget(self.right_list)
        middle_layout.addLayout(right_panel)

        self.main_layout.addLayout(middle_layout)

        # Bottom: X variable selector
        x_selector_layout = QHBoxLayout()
        self.x_selector = QComboBox()
        x_selector_layout.addWidget(QLabel("Loaded statistics:"))
        x_selector_layout.addWidget(self.x_selector)
        self.main_layout.addLayout(x_selector_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect signals
        self.band_selector.currentIndexChanged.connect(self.update_masks)
        self.add_left_button.clicked.connect(lambda: self.add_to_axis("left"))
        self.add_right_button.clicked.connect(lambda: self.add_to_axis("right"))

        self.update_plot()

    def load_tomo_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select .tomo Folder")
        if folder_path and folder_path.endswith(".tomo"):
            self.scene = SceneStats.load(folder_path)
            self.band_selector.clear()
            self.band_selector.addItems(self.scene.bands)
            self.update_masks()
            self.update_variable_selectors()
            self.status_bar.showMessage(f"Loaded tomogram from {folder_path}", 3000)
        else:
            self.status_bar.showMessage("Please select a valid .tomo folder.", 3000)

    def update_masks(self):
        band = self.band_selector.currentText()
        self.mask_selector.clear()
        if self.scene and band in self.scene.bands:
            self.mask_selector.addItems(self.scene[band].masks.keys())

    def update_variable_selectors(self):
        common_cols = self.get_common_columns()
        self.x_selector.clear()
        self.left_y_selector.clear()
        self.right_y_selector.clear()
        self.x_selector.addItems(common_cols)
        self.left_y_selector.addItems(common_cols)
        self.right_y_selector.addItems(common_cols)

    def get_common_columns(self):
        all_columns = [set(df.columns) for df in self.scene.stats.values()]
        return sorted(set.intersection(*all_columns)) if all_columns else []

    def add_to_axis(self, axis):
        if not self.scene:
            self.status_bar.showMessage("No tomogram loaded.", 3000)
            return
        band = self.band_selector.currentText()
        mask = self.mask_selector.currentText()
        layer = self.layer_selector.currentText()
        x_var = self.x_selector.currentText()
        y_var = self.left_y_selector.currentText() if axis == "left" else self.right_y_selector.currentText()

        if not band or not mask or not x_var or not y_var:
            self.status_bar.showMessage("Please select band, mask, and variables.", 3000)
            return

        key = mask + "_" + layer
        if key not in self.scene.stats:
            self.status_bar.showMessage(f"No stats found for {key}.", 3000)
            return

        df = self.scene.stats[key]
        if x_var not in df.columns or y_var not in df.columns:
            self.status_bar.showMessage("Selected variables not found in data.", 3000)
            return

        x = df[x_var]
        y = df[y_var]
        label = f"{self.scene.id}: {band} (Mask: {mask}, Layer: {layer})"

        if axis == "left":
            self.left_data.append((x, y, label))
            self.left_list.addItem(QListWidgetItem(label))
        else:
            self.right_data.append((x, y, label))
            self.right_list.addItem(QListWidgetItem(label))

        self.status_bar.showMessage(f"Added {label} to {axis} axis.", 3000)
        self.update_plot()

    def update_plot(self):
        fig = Figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        for x, y, label in self.left_data:
            ax1.plot(x, y, label=label)
        for x, y, label in self.right_data:
            ax2.plot(x, y, linestyle='--', label=label)

        ax1.set_xlabel(self.x_selector.currentText() or "X Axis")
        ax1.set_ylabel("Left Y Axis")
        ax2.set_ylabel("Right Y Axis")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()

        self.canvas.figure = fig
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TomogramPlotter()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())