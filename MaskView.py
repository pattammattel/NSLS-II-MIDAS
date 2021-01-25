import sys
import tifffile as tf
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np
import os
import logging

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import ImageView, PlotWidget

from StackCalcs import *

logger = logging.getLogger()


class MaskSpecViewer(QtWidgets.QMainWindow):

    def __init__(self, xanes_stack=None, xrf_map=None, energy=[]):
        super(MaskSpecViewer, self).__init__()
        uic.loadUi('MaskedView.ui', self)

        self.xanes_stack = xanes_stack
        self.xrf_map = xrf_map
        self.energy = energy
        self.xanes_stack = tf.imread('Site4um.tiff').transpose(0,2,1)

        self.xanes_view.setImage(self.xanes_stack)
        self.xanes_view.ui.menuBtn.hide()
        self.xanes_view.ui.roiBtn.hide()
        (self.dim1, self.dim3, self.dim2) = self.xanes_stack.shape
        self.xanes_view.setPredefinedGradient('viridis')
        self.xanes_view.setCurrentIndex(self.dim1//2)

        self.xrf_map = self.xanes_stack[-1]
        self.xrf_view.setImage(self.xrf_map)
        self.xrf_view.ui.menuBtn.hide()
        self.xrf_view.ui.roiBtn.hide()
        self.xrf_view.setPredefinedGradient('bipolar')

        self.mask_view.ui.menuBtn.hide()
        self.mask_view.ui.roiBtn.hide()

        #connections
        self.sldr_xrf_low.valueChanged.connect(self.create_mask)
        self.sldr_xrf_high.valueChanged.connect(self.create_mask)
        self.pb_apply_mask.clicked.connect(self.apply_mask_to_xanes)
        self.actionLoad_Energy_List.triggered.connect(self.load_energy)

    def create_mask(self):
        self.threshold_low = np.around(self.sldr_xrf_low.value()*0.01,3)
        self.threshold_high = np.around(self.sldr_xrf_high.value() * 0.01,3)
        self.sldr_xrf_low.setMaximum(self.sldr_xrf_high.value()+1)
        self.sldr_xrf_high.setMinimum(self.sldr_xrf_low.value()+1)
        self.norm_xrf_map = remove_nan_inf(self.xrf_map)/remove_nan_inf(self.xrf_map.max())
        self.norm_xrf_map[self.norm_xrf_map<self.threshold_low] = 0
        self.norm_xrf_map[self.norm_xrf_map > self.threshold_high] = 0
        self.xrf_view.setImage(self.norm_xrf_map)
        self.le_sldr_vals.setText(str(self.threshold_low)+' to '+str(self.threshold_high))
        self.statusbar.showMessage('New Threshold Applied')
        self.xrf_mask  = np.where(self.norm_xrf_map > 0 , self.norm_xrf_map, 0)
        self.xrf_mask[self.xrf_mask>0] = 1
        self.mask_view.setImage(self.xrf_mask)

    def load_energy(self):
        """To load energy list that will be used for plotting the spectra.
        number of stack should match length of energy list"""

        file_name = QFileDialog().getOpenFileName(self, "Open energy list", '', 'text file (*.txt)')

        try:
            self.energy = np.loadtxt(str(file_name[0]))
            logger.info ('Energy file loaded')
            assert len(self.energy) == self.dim1

        except OSError:
            logger.error("No File selected")
            pass


    def apply_mask_to_xanes(self):

        """Generates a mask with 0 and 1 from the choosen threshold and multply with the xanes stack.
        A spectrum will be generated from the new masked stack """

        self.masked_xanes = self.xanes_stack*self.xrf_mask
        self.xanes_view.setImage(self.masked_xanes)
        self.xanes_view.setCurrentIndex(self.dim1 // 2)
        self.statusbar.showMessage('Mask Applied to XANES')
        self.mask_spec = get_mean_spectra(self.masked_xanes)

        if len(self.energy) != 0:
            self.xdata = self.energy
        else:
            self.xdata = np.arange(0,self.dim1)
            self.statusbar.showMessage('No Energy List Available; Integer values are used for plotting')

        self.spectrum_view.plot(self.xdata, self.mask_spec, clear=True)

    def import_a_mask(self):
        self.statusbar.showMessage('A New Mask Imported')
        pass

    def export_mask(self):
        self.statusbar.showMessage('Mask Exported')
        pass


if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(stream_handler)

    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = MaskSpecViewer()
    window.show()
    sys.exit(app.exec_())



