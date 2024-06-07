from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QColorDialog,
    QComboBox
)
from PyQt5.QtGui import QPalette, QColor

import sys

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.color_bind = {}
        self.component_bind = {}

        self.setWindowTitle("My App")
        self.setFixedSize(QSize(800, 600))

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(True)
        
        hbox = QHBoxLayout()
        btn = QPushButton("choose color", self)
        hbox.addWidget(btn)
        btn.clicked.connect(self.choose_color)
        label = QLabel("  ")
        hbox.addWidget(label)
        self.component_bind["iosram"] = label
        cmb = QComboBox()
        cmb.addItem("iosram")
        cmb.addItem("rf")
        cmb.addItem("dpu")
        hbox.addWidget(cmb)
        widget = QWidget()
        widget.setLayout(hbox)
        tabs.addTab(widget, "Resources")

        for n, color in enumerate(["red", "green", "blue", "yellow"]):
            tabs.addTab(Color(color), color)

        self.setCentralWidget(tabs)
        self.show()
    
    @pyqtSlot()
    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            print(color.name())
            self.setStyleSheet(f"background-color: {color.name()}")


app = QApplication(sys.argv)

window = MainWindow()

sys.exit(app.exec_())