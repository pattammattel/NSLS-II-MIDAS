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


class StackSpecViewer(QtWidgets.QMainWindow):

    def __init__(self, im_stack):
        super(StackSpecViewer, self).__init__()

        uic.loadUi('StackViewer.ui', self)
        self.im_stack = im_stack
        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape
        self.x_energy = np.arange(0,self.dim1)
        self.image_view.setImage(self.im_stack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')
        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1 * 0.05)
        self.image_view.setCurrentIndex(self.stack_center)

        '''
        self.image_roi = pg.ROI(
            pos=(int(self.dim2 // 2), int(self.dim3 // 2)),
            size=(int(self.dim2 * 0.1), int(self.dim3 * 0.1)),
            scaleSnap=True, translateSnap=True, rotateSnap=True
        )
        
        '''
        self.cn = int(self.dim2 // 2)
        self.sz = np.max([int(self.dim2 * 0.15),int(self.dim3 * 0.15)])
        self.image_roi = pg.PolyLineROI([[0,0], [0,self.sz], [self.sz,self.sz], [self.sz,0]],
                                        pos =(int(self.dim2 // 2), int(self.dim3 // 2)), closed=True)

        self.image_view.addItem(self.image_roi)
        self.spec_roi = pg.LinearRegionItem(values=(self.stack_center - self.stack_width,
                                                    self.stack_center + self.stack_width))
        self.spec_roi.setBounds([0, self.dim1])
        self.sb_roi_spec_s.setValue(self.stack_center - self.stack_width)
        self.sb_roi_spec_e.setValue(self.stack_center + self.stack_width)
        self.reset_image()
        self.update_spectrum()
        self.update_image_roi()

        # connections
        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_log_view.clicked.connect(self.update_image)
        self.pb_reset.clicked.connect(self.reset_image)
        self.sb_roi_spec_s.valueChanged.connect(self.set_spec_roi)
        self.sb_roi_spec_e.valueChanged.connect(self.set_spec_roi)
        # self.pb_play_stack.clicked.connect(self.play_stack)

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

        # print(self.updated_im_stack[:, xmax, ymax])
        self.xdata = np.arange(0, self.dim1, 1)

        ydata = remove_nan_inf(get_sum_spectra(self.updated_im_stack[:, xmin:xmax, ymin:ymax]))
        self.spectrum_view.plot(self.x_energy, ydata, clear=True)
        self.spectrum_view.addItem(self.spec_roi)

    def update_image_roi(self):
        self.spec_lo, self.spec_hi = self.spec_roi.getRegion()
        self.sb_roi_spec_s.setValue(int(self.spec_lo))
        self.sb_roi_spec_e.setValue(int(self.spec_hi))
        self.le_roi_spec_size.setText(str(int(self.spec_hi - self.spec_lo)))
        self.image_view.setImage(self.updated_im_stack[int(self.spec_lo):int(self.spec_hi), :, :].mean(0))

    def set_spec_roi(self):
        if self.sync_spec_roi.isChecked():
            self.spec_lo_, self.spec_hi_ = int(self.sb_roi_spec_s.value()), int(self.sb_roi_spec_e.value())
            self.spec_roi.setRegion((self.spec_lo_, self.spec_hi_))
        else:
            pass

    def play_stack(self):
        self.image_view.play(rate=5)

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
        self.hs_comp_number.setMaximum(self.dim1 - 1)

        self.image_view.setImage(self.comp_stack)
        self.image_view.setPredefinedGradient('viridis')
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.image_view2.setImage(self.decomp_map)
        self.image_view2.setPredefinedGradient('bipolar')
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()

        # connection
        self.update_image()
        self.pb_show_all.clicked.connect(self.show_all_spec)
        self.hs_comp_number.valueChanged.connect(self.update_image)
        self.actionSave.triggered.connect(self.save_comp_data)

    def update_image(self):
        im_index = self.hs_comp_number.value()
        self.spectrum_view.plot(self.decon_spectra[:, im_index], clear=True)
        self.component_view.plot(self.comp_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])

    def show_all_spec(self):
        self.spectrum_view.clear()
        plt_clrs = ['g', 'r', 'c', 'm', 'y', 'w'] * 2
        offsets = np.arange(0, 2, 0.2)
        self.spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.spectrum_view.plot((self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                                    pen=plt_clrs[ii], name="component" + str(ii + 1))

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        tf.imsave(str(file_name[0]) + '_components.tiff', np.float32(self.comp_stack.transpose(0, 2, 1)), imagej=True)
        plt.imsave(str(file_name[0]) + '_component_map.png', np.float32(self.decomp_map.T))
        np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)
        np.savetxt(str(file_name[0]) + '_component_spec.txt', self.comp_spectra)

    # add energy column


class ClusterViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_images, X_cluster, decon_spectra):
        super(ClusterViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('ClusterView.ui', self)

        self.decon_images = decon_images
        self.X_cluster = X_cluster
        self.decon_spectra = decon_spectra
        (self.dim1, self.dim3, self.dim2) = self.decon_images.shape
        self.hs_comp_number.setMaximum(self.dim1 - 1)
        self.X_cluster = X_cluster

        self.image_view.setImage(self.decon_images, autoHistogramRange=True, autoLevels=True)
        self.image_view.setPredefinedGradient('viridis')
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.cluster_view.setImage(self.X_cluster, autoHistogramRange=True, autoLevels=True)
        self.cluster_view.setPredefinedGradient('bipolar')
        self.cluster_view.ui.histogram.hide()
        self.cluster_view.ui.menuBtn.hide()
        self.cluster_view.ui.roiBtn.hide()

        # connection
        self.update()
        self.hs_comp_number.valueChanged.connect(self.update)
        self.actionSave.triggered.connect(self.save_clust_data)

    def update(self):
        im_index = self.hs_comp_number.value()
        self.component_view.plot(self.decon_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.decon_images[im_index])

    def save_clust_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        tf.imsave(str(file_name[0]) + '_cluster.tiff', np.float32(self.decon_images.transpose(0, 2, 1)), imagej=True)
        plt.imsave(str(file_name[0]) + '_cluster_map.png', np.float32(self.X_cluster.T))
        np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)


class XANESViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_ims, im_stack, e_list, refs):
        super(XANESViewer, self).__init__()

        uic.loadUi('XANESViewer.ui', self)
        self.decon_ims = decon_ims
        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs

        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape
        self.cn = int(self.dim2 // 2)
        self.sz = np.max([int(self.dim2 * 0.25),int(self.dim3 * 0.25)])
        self.image_roi = pg.PolyLineROI([[0,0], [0,self.sz], [self.sz,self.sz], [self.sz,0]],
                                        pos =(int(self.dim2 // 2), int(self.dim3 // 2)), closed=True)
        self.image_roi.addTranslateHandle([self.sz//2, self.sz//2], [2, 2])
        self.image_view.setImage(self.im_stack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')
        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1 * 0.05)
        self.image_view.setCurrentIndex(self.stack_center)
        self.image_view.addItem(self.image_roi)
        self.xdata = self.e_list + self.sb_e_shift.value()

        self.display_all_data()

        self.update_spectrum()
        # connections
        self.sb_e_shift.valueChanged.connect(self.update_spectrum)
        self.sb_e_shift.valueChanged.connect(self.re_fit_xanes)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_save_chem_map.clicked.connect(self.save_chem_map)
        #self.pb_save_spe_fit.clicked.connect(self.reset_roi)
        self.pb_save_spe_fit.clicked.connect(self.save_spec_fit)
        # self.pb_play_stack.clicked.connect(self.play_stack)

    def display_all_data(self):
        self.image_view_maps.setImage(self.decon_ims)
        self.image_view_maps.setPredefinedGradient('bipolar')
        self.image_view_maps.ui.menuBtn.hide()
        self.image_view_maps.ui.roiBtn.hide()
        new_ref = interploate_E(self.refs, self.xdata)

        plt_clrs = ['c', 'm', 'y', 'w']*2
        self.spectrum_view_refs.addLegend()
        for ii in range(new_ref.shape[0]):
            self.spectrum_view_refs.plot(self.xdata, new_ref[ii], pen=plt_clrs[ii], name="ref" + str(ii + 1))

    def update_spectrum(self):

        self.roi_img = self.image_roi.getArrayRegion(self.im_stack, self.image_view.imageItem, axes=(1, 2))
        sizex, sizey = self.roi_img.shape[1], self.roi_img.shape[2]
        posx, posy = self.image_roi.pos()
        self.le_roi_xs.setText(str(int(posx))+':' +str(int(posy)))
        self.le_roi_xe.setText(str(sizex) +','+ str(sizey))

        self.xdata1 = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_sum_spectra(self.roi_img)
        new_ref = interploate_E(self.refs, self.xdata1)
        coeffs, r = opt.nnls(new_ref.T, self.ydata1)
        self.fit_ = np.dot(coeffs, new_ref)
        pen = pg.mkPen('g', width=1.5)
        pen2 = pg.mkPen('r', width=1.5)
        self.spectrum_view.addLegend()
        self.spectrum_view.plot(self.xdata1, self.ydata1, pen=pen, name="Data", clear=True)
        self.spectrum_view.plot(self.xdata1, self.fit_, name="Fit", pen=pen2)
        self.le_r_sq.setText(str(np.around(r / self.ydata1.sum(), 4)))

    def re_fit_xanes(self):
        self.decon_ims = xanes_fitting(self.im_stack, self.e_list + self.sb_e_shift.value(), self.refs, method='NNLS')
        self.image_view_maps.setImage(self.decon_ims.T)

    def save_chem_map(self):
        file_name = QFileDialog().getSaveFileName(self, "save image", '', 'image data (*tiff)')
        try:
            tf.imsave(str(file_name[0]) + '.tiff', np.float32(self.decon_ims), imagej=True)
        except:
            logger.error('No file to save')
            pass

    def save_spec_fit(self):
        try:
            to_save = np.column_stack((self.xdata1, self.ydata1, self.fit_))
            file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectrum and fit (*txt)')
            np.savetxt(str(file_name[0]) + '.txt', to_save)
        except:
            logger.error('No file to save')
            pass


    '''
    def display_rgb(self):
        self.image_view_maps.clear()
        clrs = ['r','g','k']
        for ii in range(3):
            self.image_view_maps.addItem(self.im_stack[ii])
            self.image_view_maps.setPredefinedGradient('thermal')
    

    def reset_roi(self):
        self.image_view.removeItem(self.image_roi)
        self.image_roi = pg.PolyLineROI([[0,0], [0,self.sz], [self.sz,self.sz], [self.sz,0]],
                                        pos =(int(self.dim2 // 2), int(self.dim3 // 2)), closed=True)
        self.image_roi.addRotateHandle([self.sz // 2, self.sz // 2], [2, 2])

    '''

class ScatterPlot(QtWidgets.QMainWindow):

    def __init__(self, img1, img2):
        super(ScatterPlot, self).__init__()

        uic.loadUi('ScatterView.ui', self)
        w1 = self.scatterViewer.addPlot()
        self.img1 = img1.flatten()
        self.img2 = img2.flatten()
        s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 120))
        s1.setData(self.img1,self.img2)
        w1.addItem(s1)

class ScatterPlot2(QtWidgets.QMainWindow):

    def __init__(self, img1, img2):
        super(ScatterPlot2, self).__init__()

        uic.loadUi('ScatterView.ui', self)
        w1 = self.scatterViewer.addPlot()
        self.img1 = img1.flatten()
        self.img2 = img2.flatten()
        s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 120))
        s1.setData(self.img1,self.img2)
        w1.addItem(s1)





