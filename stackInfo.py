import os, sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import  QFileDialog, QApplication

ui_path = os.path.dirname(os.path.abspath(__file__))

class StackInfo(QtWidgets.QMainWindow):

    def __init__(self, text_to_write :str = ' '):
        super(StackInfo, self).__init__()
        uic.loadUi(os.path.join(ui_path, 'uis/log.ui'), self)

        self.text_to_write = text_to_write
        self.pte_run_cmd.setPlainText(self.text_to_write)

        #connections
        self.pb_save_cmd.clicked.connect(self.save_file)
        self.pb_clear_cmd.clicked.connect(self.clear_text)

    def save_file(self):
        S__File = QFileDialog.getSaveFileName(None, 'SaveFile', '/', "txt Files (*.txt)")

        Text = self.pte_run_cmd.toPlainText()
        if S__File[0]:
            with open(S__File[0], 'w') as file:
                file.write(Text)

    def clear_text(self):
        self.pte_run_cmd.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StackInfo()
    window.show()
    sys.exit(app.exec_())