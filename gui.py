import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QPushButton
from PyQt5.QtWidgets import QLineEdit, QFileDialog, QMainWindow, QLabel, QProgressBar
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sign_detection import sliding_window, draw_window

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Traffic Sign Detection"
        self.left = 300
        self.top = 100
        self.width = 800
        self.height = 530
        self.image = None
        self.initGUI()

    def initGUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("border: 2px solid #808080; background-color:#C0C0C0")

        self.titleLabel = QLabel(self)
        self.titleLabel.setText("Traffic Sign Detection Application")
        self.titleLabel.setStyleSheet("border: 0px; font-size:20pt; font-family: Times New Roman; color: #2C67B7")
        self.titleLabel.setGeometry(200, 20, 400, 50)

        self.label = QLabel(self)
        self.label.setGeometry(100, 80, 600, 300)

        self.uploadBtn = QPushButton("Select Image", self)
        self.uploadBtn.setGeometry(240, 400, 120, 40)
        self.uploadBtn.setStyleSheet("background-color:#4e4e4e; color:#f7f7f7;")
        self.uploadBtn.clicked.connect(self.uploadImage)

        self.detectBtn = QPushButton("Detect Traffic Sign", self)
        self.detectBtn.setGeometry(400, 400, 120, 40)
        self.detectBtn.setStyleSheet("background-color:#4e4e4e; color:#f7f7f7")
        self.detectBtn.clicked.connect(self.detectSign)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(100, 460, 600, 25)
        self.pbar.setStyleSheet("border: 0px;")
        self.pbar.setAlignment(QtCore.Qt.AlignCenter)

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*png *jpg)", options=options)
        if fileName:
            return fileName

    def uploadImage(self):
        file = self.openFileNameDialog()
        pixmap = QPixmap(file)
        pixmap = pixmap.scaled(600, 300)
        self.image = file
        self.label.setPixmap(pixmap)
        print(int(file.split("/")[-1].split("_")[0]))

    def detectSign(self):
        img = self.image
        print(img)
        svc_data = "svc_data.pkl"
        print(svc_data)
        if img:
            predictions = sliding_window(img, svc_data, self.pbar)
            print(predictions)
            for p in predictions:
                print("Prediction:", p["Prediction"], "Prob:", p["Prob"], "x1:", p['x1'], "y1:", p['y1'],
                      "x2:", p['x2'], "y2:", p['y2'], "height:", p["height"], "width:", p["width"])

                draw_window(img, p['x1'], p['y1'], p['x2'], p['y2'], p['width'],
                            p['height'], p['Prediction'], p['Prob'])

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()