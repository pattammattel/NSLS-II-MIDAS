import sys
import tifffile as tf
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import ImageView, PlotWidget
import numpy as np
import os
from StackCalcs import *

colors = [
    (68, 1, 84),
    (58, 82, 139),
    (32, 144, 140),
    (94, 201, 97),
    (253, 231, 36),
]

colors2 = [
    (0, 255, 255),
    (0, 0, 255),
    (0, 0, 0),
    (255, 0, 0),
    (255, 255, 0),
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors)
cmap2 = pg.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors2)
#pg.setConfigOption('background', 'g')


class StackSpecViewer(QtWidgets.QMainWindow):

    def __init__(self, im_stack):
        super(StackSpecViewer, self).__init__()

        uic.loadUi('StackViewer.ui', self)
        self.im_stack = im_stack
        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape

        self.image_view.setImage(self.im_stack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setColorMap(cmap)
        self.image_view.setCurrentIndex(int(self.dim1//2))

        self.image_roi = pg.ROI(
            pos=(int(self.dim2 // 2), int(self.dim3 // 2)),
            size=(int(self.dim2 * 0.1), int(self.dim3 * 0.1)),
            scaleSnap=True, translateSnap=True, rotateSnap=True
        )

        self.image_view.addItem(self.image_roi)
        self.spec_roi = pg.LinearRegionItem(values=(0, 10))
        self.reset_image()
        self.update_spectrum()
        self.update_image_roi()


        #connections
        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_log_view.clicked.connect(self.update_image)
        self.pb_reset.clicked.connect(self.reset_image)
        #self.pb_play_stack.clicked.connect(self.play_stack)

    def update_spectrum(self):
        # Obtaining coordinates of ROI graphic in the image plot
        self.image_coord_handles = self.image_roi.getState()
        self.posimage = self.image_coord_handles['pos']
        self.sizeimage = self.image_coord_handles['size']

        posx = int(self.posimage[0])
        sizex = int(self.sizeimage[0])
        posy = int(self.posimage[1])
        sizey = int(self.sizeimage[1])
        xmin = posx
        xmax = posx + sizex
        ymin = posy
        ymax = posy + sizey

        self.le_roi_xs.setText(str(xmin))
        self.le_roi_xe.setText(str(xmax))
        self.le_roi_ys.setText(str(ymin))
        self.le_roi_ye.setText(str(ymax))
        self.le_roi_size.setText(str(self.sizeimage))

        #print(self.updated_im_stack[:, xmax, ymax])
        xdata = np.arange(0, self.dim1, 1)
        ydata = remove_nan_inf(get_sum_spectra(self.updated_im_stack[:, xmin:xmax,ymin:ymax]))
        self.spectrum_view.plot(xdata, ydata, clear=True)
        self.spectrum_view.addItem(self.spec_roi)

    def update_image_roi(self):
        self.spec_lo, self.spec_hi = self.spec_roi.getRegion()
        self.le_roi_spec_s.setText(str(int(self.spec_lo)))
        self.le_roi_spec_e.setText(str(int(self.spec_hi)))
        self.le_roi_spec_size.setText(str(int(self.spec_hi-self.spec_lo)))
        self.image_view.setImage(self.updated_im_stack[int(self.spec_lo):int(self.spec_hi), :, :].mean(0))

    def play_stack(self):
        self.image_view.play(rate = 5)

    def update_image(self):
        self.updated_im_stack = remove_nan_inf(np.log(self.im_stack))
        self.update_spectrum()
        self.update_image_roi()

    def reset_image(self):
        self.updated_im_stack = remove_nan_inf(self.im_stack)
        self.update_spectrum()
        self.update_image_roi()


class ComponentViewer(QtWidgets.QMainWindow):

    def __init__(self, comp_stack, comp_spectra, decon_spectra):
        super(ComponentViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('ComponentView.ui', self)

        self.comp_stack = comp_stack
        self.comp_spectra = comp_spectra
        self.decon_spectra = decon_spectra
        (self.dim1, self.dim3, self.dim2) = self.comp_stack.shape
        self.hs_comp_number.setMaximum(self.dim1-1)

        self.image_view.setImage(self.comp_stack,autoHistogramRange=True,autoLevels = True)
        self.image_view.setColorMap(cmap)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        #connection
        self.update_image()
        self.hs_comp_number.valueChanged.connect(self.update_image)

    def update_image(self):
        im_index = self.hs_comp_number.value()
        self.spectrum_view.plot(self.decon_spectra[:, im_index], clear=True)
        self.component_view.plot(self.comp_spectra[:, im_index], clear=True)
        #self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])


class ClusterViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_images,X_cluster, decon_spectra):
        super(ClusterViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('ClusterView.ui', self)

        self.decon_images = decon_images
        self.X_cluster = X_cluster
        self.decon_spectra = decon_spectra
        (self.dim1, self.dim3, self.dim2) = self.decon_images.shape
        self.hs_comp_number.setMaximum(self.dim1-1)
        self.X_cluster = X_cluster

        self.image_view.setImage(self.decon_images,autoHistogramRange=True,autoLevels = True)
        self.image_view.setColorMap(cmap)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.cluster_view.setImage(self.X_cluster, autoHistogramRange=True,autoLevels = True)
        self.cluster_view.setColorMap(cmap2)
        self.cluster_view.ui.histogram.hide()
        self.cluster_view.ui.menuBtn.hide()
        self.cluster_view.ui.roiBtn.hide()

        #connection
        self.update()
        self.hs_comp_number.valueChanged.connect(self.update)

    def update(self):
        im_index = self.hs_comp_number.value()
        self.component_view.plot(self.decon_spectra[:, im_index], clear=True)
        #self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.decon_images[im_index])


#im_ = tf.imread(r'C:\Users\pattammattel\Desktop\Spectromicroscopy\python_codes\HXN_Data\Site4um.tiff').transpose(0,2,1)
'''

def main(updated_im_stack):
    app = QtWidgets.QApplication(sys.argv)
    main = StackSpecViewer(updated_im_stack)
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(im_)
'''