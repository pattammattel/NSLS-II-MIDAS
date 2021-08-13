import os
import tifffile as tf

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDesktopWidget, QApplication, QSizePolicy
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
import sys

def sumNumbers(a, b):
    result = a + b
    return result


class SumFunction(QtWidgets.QMainWindow):

    def __init__(self):
        super(SumFunction, self).__init__()
        uic.loadUi('test.ui', self)

        self.pb_test.clicked.connect(self.displaySum)
        self.actionLoad_Image.triggered.connect(self.displayImage)

    def displaySum(self):
        inp1 = self.dsb_input1.value()
        inp2 = self.dsb_input2.value()
        result = sumNumbers(inp1, inp2)
        self.label_result.setText(f'Result:{result}')

    def displayImage(self):
        file_name = QFileDialog().getOpenFileName(self, "Load An Image", '', 'tiff file (*.tiff *.tif)')
        image = tf.imread(file_name[0])
        self.image_viewer.setImage(image)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = SumFunction()
    window.show()
    sys.exit(app.exec_())
