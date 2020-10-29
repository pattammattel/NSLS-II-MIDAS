import sys
import tifffile as tf
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import ImageView, PlotWidget
import numpy as np
import os
from StackCalcs import *

class StackSpecViewer(QtWidgets.QMainWindow):

    def __init__(self, im_stack):
        super(StackSpecViewer, self).__init__()

        uic.loadUi('StackViewer.ui', self)
        self.im_stack = im_stack
        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape

        self.image_view.setImage(self.im_stack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')
        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1*0.05)
        self.image_view.setCurrentIndex(self.stack_center)

        self.image_roi = pg.ROI(
            pos=(int(self.dim2 // 2), int(self.dim3 // 2)),
            size=(int(self.dim2 * 0.1), int(self.dim3 * 0.1)),
            scaleSnap=True, translateSnap=True, rotateSnap=True
        )

        self.image_view.addItem(self.image_roi)
        self.spec_roi = pg.LinearRegionItem(values=(self.stack_center-self.stack_width,
                                                    self.stack_center+self.stack_width))
        self.spec_roi.setBounds([0, self.dim1])
        self.sb_roi_spec_s.setValue(self.stack_center-self.stack_width)
        self.sb_roi_spec_e.setValue(self.stack_center + self.stack_width)
        self.reset_image()
        self.update_spectrum()
        self.update_image_roi()


        #connections
        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_log_view.clicked.connect(self.update_image)
        self.pb_reset.clicked.connect(self.reset_image)
        self.sb_roi_spec_s.valueChanged.connect(self.set_spec_roi)
        self.sb_roi_spec_e.valueChanged.connect(self.set_spec_roi)
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
        self.sb_roi_spec_s.setValue(int(self.spec_lo))
        self.sb_roi_spec_e.setValue(int(self.spec_hi))
        self.le_roi_spec_size.setText(str(int(self.spec_hi-self.spec_lo)))
        self.image_view.setImage(self.updated_im_stack[int(self.spec_lo):int(self.spec_hi), :, :].mean(0))

    def set_spec_roi(self):
        if self.sync_spec_roi.isChecked():
            self.spec_lo_, self.spec_hi_ = int(self.sb_roi_spec_s.value()), int(self.sb_roi_spec_e.value())
            self.spec_roi.setRegion((self.spec_lo_, self.spec_hi_))
        else:
            pass

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

    def __init__(self, comp_stack, comp_spectra, decon_spectra, decomp_map):
        super(ComponentViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('ComponentView.ui', self)

        self.comp_stack = comp_stack
        self.comp_spectra = comp_spectra
        self.decon_spectra = decon_spectra
        self.decomp_map = decomp_map

        (self.dim1, self.dim3, self.dim2) = self.comp_stack.shape
        self.hs_comp_number.setMaximum(self.dim1-1)

        self.image_view.setImage(self.comp_stack)
        self.image_view.setPredefinedGradient('viridis')
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.image_view2.setImage(self.decomp_map)
        self.image_view2.setPredefinedGradient('bipolar')
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()

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
        self.image_view.setPredefinedGradient('viridis')
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.cluster_view.setImage(self.X_cluster, autoHistogramRange=True,autoLevels = True)
        self.cluster_view.setPredefinedGradient('bipolar')
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


class XANESViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_ims, im_stack, e_list, refs):
        super(XANESViewer, self).__init__()

        uic.loadUi('XANESViewer.ui', self)
        self.decon_ims = decon_ims
        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs


        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape
        self.image_roi = pg.ROI(
            pos=(int(self.dim2 // 2), int(self.dim3 // 2)),
            size=(int(self.dim2 * 0.1), int(self.dim3 * 0.1)),
            scaleSnap=True, translateSnap=True, rotateSnap=True
        )
        self.image_view.setImage(self.im_stack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')
        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1*0.05)
        self.image_view.setCurrentIndex(self.stack_center)
        self.image_view.addItem(self.image_roi)
        self.xdata = self.e_list + self.sb_e_shift.value()

        self.display_all_data()

        self.update_spectrum()
        #connections
        self.sb_e_shift.valueChanged.connect(self.update_spectrum)
        self.sb_e_shift.valueChanged.connect(self.re_fit_xanes)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        #self.pb_play_stack.clicked.connect(self.play_stack)

    def display_all_data(self):

        self.image_view_maps.setImage(self.decon_ims)
        self.image_view_maps.setPredefinedGradient('bipolar')
        self.image_view_maps.ui.menuBtn.hide()
        self.image_view_maps.ui.roiBtn.hide()
        new_ref = interploate_E(self.refs, self.xdata)

        plt_clrs = ['c','m','y']
        self.spectrum_view_refs.addLegend()
        for ii in range(new_ref.shape[0]):
            self.spectrum_view_refs.plot(self.xdata, new_ref[ii], pen = plt_clrs[ii], name = "ref"+str(ii))

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

        # print(self.updated_im_stack[:, xmax, ymax])
        xdata = self.e_list+self.sb_e_shift.value()
        ydata1 = remove_nan_inf(get_sum_spectra(self.im_stack[:, xmin:xmax, ymin:ymax]))
        new_ref = interploate_E(self.refs,xdata)
        coeffs,r = opt.nnls(new_ref.T,ydata1)
        fit_ = np.dot(coeffs,new_ref)
        pen = pg.mkPen('g', width=1.5)
        pen2 = pg.mkPen('r', width=1.5)
        self.spectrum_view.addLegend()
        self.spectrum_view.plot(xdata, ydata1, pen=pen, name ="Data", clear=True)
        self.spectrum_view.plot(xdata, fit_, name = "Fit", pen=pen2)
        self.le_r_sq.setText(str(np.around(r/ydata1.sum(), 4)))

    def re_fit_xanes(self):
        map = xanes_fitting(self.im_stack , self.e_list+self.sb_e_shift.value(), self.refs, method='NNLS')
        self.image_view_maps.setImage(map.T)

