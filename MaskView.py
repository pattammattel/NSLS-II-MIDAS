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
        uic.loadUi('uis/MaskedView.ui', self)

        self.xanes_stack = xanes_stack
        self.xrf_map = xrf_map
        self.energy = energy
        self.xrf_map = self.xanes_stack[-1]
        self.view_data()

        #connections
        self.sldr_xrf_low.valueChanged.connect(self.create_mask)
        self.sldr_xrf_high.valueChanged.connect(self.create_mask)
        self.pb_apply_mask.clicked.connect(self.apply_mask_to_xanes)
        self.pb_export_mask.clicked.connect(self.export_mask)
        self.pb_import_mask.clicked.connect(self.import_a_mask)
        self.actionLoad_Energy_List.triggered.connect(self.load_energy)
        self.actionLoad_XANES_Stack.triggered.connect(self.load_xanes_stack)
        self.actionLoad_XRF_Map.triggered.connect(self.load_xrf_map)

    def view_data(self):

        self.xanes_view.setImage(self.xanes_stack)
        self.xanes_view.ui.menuBtn.hide()
        self.xanes_view.ui.roiBtn.hide()
        (self.dim1, self.dim3, self.dim2) = self.xanes_stack.shape
        self.xanes_view.setPredefinedGradient('viridis')
        self.xanes_view.setCurrentIndex(self.dim1//2)
        self.statusbar.showMessage('One image from the XANES stack is used as mask')
        self.xrf_view.setImage(self.xrf_map)
        self.xrf_view.ui.menuBtn.hide()
        self.xrf_view.ui.roiBtn.hide()
        self.xrf_view.setPredefinedGradient('bipolar')

        self.mask_view.ui.menuBtn.hide()
        self.mask_view.ui.roiBtn.hide()


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

    def load_xanes_stack(self):
        """loading a new xanes stack"""
        filename = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*tiff *tif )')
        self.file_name = (str(filename[0]))
        self.xanes_stack = tf.imread(self.file_name).transpose(0,2,1)
        self.view_data()

    def load_energy(self):
        """To load energy list that will be used for plotting the spectra.
        number of stack should match length of energy list"""

        file_name = QFileDialog().getOpenFileName(self, "Open energy list", '', 'text file (*.txt)')

        try:
            self.energy = np.loadtxt(str(file_name[0]))
            logger.info ('Energy file loaded')
            assert len(self.energy) == self.dim1
            self.view_data()

        except OSError:
            logger.error("No File selected")
            pass

    def load_xrf_map(self):
        """To xrf map for masking. If 3D mean will be taken"""

        filename = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*tiff *tif )')
        self.xrf_file_name = (str(filename[0]))
        self.xrf_map = tf.imread(self.xrf_file_name)
        if self.xrf_map.ndim == 3:
            self.xrf_map = self.xrf_map.mean(0).T

        else:
            self.xrf_map = self.xrf_map.T

        assert (self.dim3,self.dim2) == self.xrf_map.shape, \
            f'Unexpected image dimensions: {self.xrf_map.shape} vs {(self.dim2,self.dim3)}'

        self.view_data()
        self.create_mask()


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
        filename = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*tiff *tif )')
        xrf_file_name = (str(filename[0]))
        self.xrf_mask = tf.imread(xrf_file_name).T
        self.statusbar.showMessage('A New Mask Imported')
        self.mask_view.setImage(self.xrf_mask)
        self.apply_mask_to_xanes()


    def export_mask(self):
        try:
            file_name = QFileDialog().getSaveFileName(self, "Save image data", '', 'image file(*tiff *tif )')
            tf.imsave(str(file_name[0]) + '.tiff', self.xrf_mask.T)
            logger.info(f'Updated Image Saved: {str(file_name[0])}')
            self.statusbar.showMessage('Mask Exported')

        except:
            logger.error('No file to save')
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



