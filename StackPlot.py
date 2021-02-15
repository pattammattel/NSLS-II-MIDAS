import sys
import tifffile as tf
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import os
import logging

from PyQt5 import QtWidgets,QtCore,QtGui, uic, QtTest
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph import ImageView, PlotWidget
from PyQt5.QtCore import pyqtSignal

#Custom .py file contains image calculations
from StackCalcs import *

logger = logging.getLogger()


class ComponentViewer(QtWidgets.QMainWindow):

    def __init__(self,  comp_stack, energy, comp_spectra, decon_spectra, decomp_map):
        super(ComponentViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('uis/ComponentView.ui', self)

        self.comp_stack = comp_stack
        self.energy = energy
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
        self.spectrum_view.setLabel('bottom','Energy')
        self.spectrum_view.setLabel('left', 'Intensity', 'A.U.')
        self.spectrum_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        self.component_view.setLabel('bottom','Energy')
        self.component_view.setLabel('left', 'Weight', 'A.U.')
        self.component_view.plot(self.energy,self.comp_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])

    def show_all_spec(self):
        self.spectrum_view.clear()
        self.plt_colors = ['g', 'r', 'c', 'm', 'y', 'w'] * 2
        offsets = np.arange(0, 2, 0.2)
        self.spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.spectrum_view.plot(self.energy,(self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                                    pen=self.plt_colors[ii], name="component" + str(ii + 1))

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        tf.imsave(str(file_name[0]) + '_components.tiff', np.float32(self.comp_stack.transpose(0, 2, 1)), imagej=True)
        tf.imsave(str(file_name[0]) + '_component_masks.tiff', np.float32(self.decomp_map.T),imagej=True)
        np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)
        np.savetxt(str(file_name[0]) + '_component_spec.txt', self.comp_spectra)

    # add energy column


class ClusterViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_images, energy, X_cluster, decon_spectra):
        super(ClusterViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('uis/ClusterView.ui', self)

        self.decon_images = decon_images
        self.energy = energy
        self.X_cluster = X_cluster
        self.decon_spectra = decon_spectra
        (self.dim1, self.dim3, self.dim2) = self.decon_images.shape
        self.hsb_cluster_number.setMaximum(self.dim1 - 1)
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
        self.update_display()
        self.hsb_cluster_number.valueChanged.connect(self.update_display)
        self.actionSave.triggered.connect(self.save_clust_data)

    def update_display(self):
        im_index = self.hsb_cluster_number.value()
        self.component_view.setLabel('bottom','Energy')
        self.component_view.setLabel('left', 'Intensity', 'A.U.')
        self.component_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.decon_images[im_index])

    def save_clust_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        if file_name[0]:

            tf.imsave(str(file_name[0]) + '_cluster.tiff', np.float32(self.decon_images.transpose(0, 2, 1)), imagej=True)
            tf.imsave(str(file_name[0]) + '_cluster_map.tiff', np.float32(self.X_cluster.T),imagej=True)
            np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)

        else:
            logger.error("Saving Cancelled")
            self.statusbar.showMessage("Saving Cancelled")
            pass


class XANESViewer(QtWidgets.QMainWindow):

    def __init__(self, im_stack, e_list, refs, ref_names):
        super(XANESViewer, self).__init__()

        uic.loadUi('uis/XANESViewer.ui', self)

        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs
        self.ref_names = ref_names
        self.selected = self.ref_names

        self.decon_ims, self.rfactor = xanes_fitting(self.im_stack, self.e_list, self.refs, method='NNLS')

        (self.dim1, self.dim3, self.dim2) = self.im_stack.shape
        self.cn = int(self.dim2 // 2)
        self.sz = np.max([int(self.dim2 * 0.15),int(self.dim3 * 0.15)])
        self.image_roi = pg.PolyLineROI([[0,0], [0,self.sz], [self.sz,self.sz], [self.sz,0]],
                                        pos =(int(self.dim2 // 2), int(self.dim3 // 2)),
                                        maxBounds=QtCore.QRect(0, 0, self.dim3, self.dim2),closed=True)
        self.image_roi.addTranslateHandle([self.sz//2, self.sz//2], [2, 2])

        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1 * 0.05)
        #self.image_view.setCurrentIndex(self.stack_center)

        self.image_view.addItem(self.image_roi)
        self.xdata = self.e_list + self.sb_e_shift.value()

        self.scrollBar_setup()
        self.display_image_data()
        self.display_references()
        self.update_spectrum()
        # connections
        self.sb_e_shift.valueChanged.connect(self.update_spectrum)
        self.sb_e_shift.valueChanged.connect(self.re_fit_xanes)
        self.pb_edit_refs.clicked.connect(self.choose_refs)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_save_chem_map.clicked.connect(self.save_chem_map)
        self.pb_save_spe_fit.clicked.connect(self.pg_export_spec_fit)
        self.hsb_xanes_stk.valueChanged.connect(self.display_image_data)
        self.hsb_chem_map.valueChanged.connect(self.display_image_data)

        #self.pb_save_spe_fit.clicked.connect(self.save_spec_fit)
        # self.pb_play_stack.clicked.connect(self.play_stack)

    def scrollBar_setup(self):
        self.hsb_xanes_stk.setValue(self.stack_center)
        self.hsb_xanes_stk.setMaximum(self.dim1 - 1)
        self.hsb_chem_map.setValue(0)
        self.hsb_chem_map.setMaximum(self.decon_ims.shape[-1]-1)

    def display_image_data(self):

        self.image_view.setImage(self.im_stack[self.hsb_xanes_stk.value()])
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')

        self.image_view_maps.setImage(self.decon_ims.transpose(2,0,1)[self.hsb_chem_map.value()])
        self.image_view_maps.setPredefinedGradient('bipolar')
        self.image_view_maps.ui.menuBtn.hide()
        self.image_view_maps.ui.roiBtn.hide()

    def display_references(self):

        self.inter_ref = interploate_E(self.refs, self.xdata)
        self.plt_colors = ['c', 'm', 'y', 'w']*4
        self.spectrum_view_refs.addLegend()
        for ii in range(self.inter_ref.shape[0]):
            if len(self.selected) != 0:
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=self.plt_colors[ii],
                                             name=self.selected[1:][ii])
            else:
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=self.plt_colors[ii],
                                             name="ref" + str(ii + 1))

    def choose_refs(self):
        'Interactively exclude some standards from the reference file'
        self.ref_edit_window = RefChooser(self.ref_names)
        self.ref_edit_window.show()
        self.ref_edit_window.signal.connect(self.update_refs)

    def update_refs(self,list_):
        self.selected = list_ # list_ is the signal from ref chooser
        self.update_spectrum()
        self.re_fit_xanes()

    def update_spectrum(self):

        self.roi_img = self.image_roi.getArrayRegion(self.im_stack, self.image_view.imageItem, axes=(1, 2))
        sizex, sizey = self.roi_img.shape[1], self.roi_img.shape[2]
        posx, posy = self.image_roi.pos()
        self.le_roi_xs.setText(str(int(posx))+':' +str(int(posy)))
        self.le_roi_xe.setText(str(sizex) +','+ str(sizey))

        self.xdata1 = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_sum_spectra(self.roi_img)
        if len(self.selected) != 0:

            self.inter_ref = interploate_E(self.refs[self.selected], self.xdata1)

        else:
            self.inter_ref = interploate_E(self.refs, self.xdata1)
        coeffs, r = opt.nnls(self.inter_ref.T, self.ydata1)
        self.fit_ = np.dot(coeffs, self.inter_ref)
        pen = pg.mkPen('g', width=1.5)
        pen2 = pg.mkPen('r', width=1.5)
        pen3 = pg.mkPen('y', width=1.5)
        self.spectrum_view.addLegend()
        self.spectrum_view.setLabel('bottom','Energy')
        self.spectrum_view.setLabel('left', 'Intensity', 'A.U.')
        self.spectrum_view.plot(self.xdata1, self.ydata1, pen=pen, name="Data", clear=True)
        self.spectrum_view.plot(self.xdata1, self.fit_, name="Fit", pen=pen2)

        for n, (coff, ref, plt_clr) in enumerate(zip(coeffs,self.inter_ref, self.plt_colors)):

            if len(self.selected) != 0:

                self.spectrum_view.plot(self.xdata1, np.dot(coff, ref), name=self.selected[1:][n],pen=plt_clr)
            else:
                self.spectrum_view.plot(self.xdata1, np.dot(coff, ref), name="ref" + str(n + 1), pen=plt_clr)

        self.le_r_sq.setText(str(np.around(r / self.ydata1.sum(), 4)))

    def re_fit_xanes(self):
        if len(self.selected) != 0:
            self.decon_ims, self.rfactor = xanes_fitting(self.im_stack, self.e_list + self.sb_e_shift.value(),
                                       self.refs[self.selected], method='NNLS')
        else:
            self.decon_ims,self.rfactor  = xanes_fitting(self.im_stack, self.e_list + self.sb_e_shift.value(),
                                       self.refs, method='NNLS')

        self.image_view_maps.setImage(self.decon_ims.transpose(2,0,1))
        self.scrollBar_setup()

    def save_chem_map(self):
        file_name = QFileDialog().getSaveFileName(self, "save image", '', 'image data (*tiff)')
        try:
            tf.imsave(str(file_name[0]) + '_xanes_map.tiff', np.float32(self.decon_ims.T), imagej=True)
            tf.imsave(str(file_name[0]) + '_rfactor.tiff', np.float32(self.rfactor.T), imagej=True)
        except:
            logger.error('No file to save')
            pass

    def save_spec_fit(self):
        try:
            to_save = np.column_stack([self.xdata1, self.ydata1, self.fit_])
            file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectrum and fit (*txt)')
            np.savetxt(str(file_name[0]) + '.txt', to_save)
        except:
            logger.error('No file to save')
            pass

    def pg_export_spec_fit(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectrum and fit (*csv)')
        exporter.export(str(file_name[0])+'.csv')

class RefChooser(QtWidgets.QMainWindow):
    signal: pyqtSignal = QtCore.pyqtSignal(list)

    def __init__(self, ref_names=[]):
        super(RefChooser, self).__init__()
        uic.loadUi('uis/RefChooser.ui', self)
        self.ref_names = ref_names
        self.all_boxes = []

        for n, i in enumerate(self.ref_names):
            self.cb_i = QtWidgets.QCheckBox(self.centralwidget)
            if n == 0:
                self.cb_i.setChecked(True)
                self.cb_i.setEnabled(False)
            self.cb_i.setObjectName(i)
            self.cb_i.setText(i)
            self.gridLayout.addWidget(self.cb_i, n, 0, 1, 1)
            self.cb_i.toggled.connect(self.enableApply)
            self.all_boxes.append(self.cb_i)

        self.pb_apply = QtWidgets.QPushButton(self.centralwidget)
        self.pb_apply.setSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed)
        self.pb_apply.setText("Apply")
        self.gridLayout.addWidget(self.pb_apply, len(self.ref_names) + 1, 0, 1, 1)
        self.pb_apply.setEnabled(False)

        self.pb_combo = QtWidgets.QPushButton(self.centralwidget)
        self.pb_combo.setText("Try All Combinations")
        self.pb_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addWidget(self.pb_combo, len(self.ref_names) + 2, 0, 1, 1)

        self.lb = QtWidgets.QLabel(self.centralwidget)
        self.lb.setText("Combo of:")
        self.lb.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addWidget(self.lb, len(self.ref_names) + 2, 1, 1, 1)

        self.sb_max_combo = QtWidgets.QSpinBox(self.centralwidget)
        self.sb_max_combo.setValue(2)
        self.sb_max_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addWidget(self.sb_max_combo, len(self.ref_names) + 2, 2, 1, 1)

        self.sb_time_delay = QtWidgets.QSpinBox(self.centralwidget)
        self.sb_time_delay.setValue(2)
        self.sb_time_delay.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addWidget(self.sb_time_delay, len(self.ref_names) + 2, 4, 1, 1)

        self.pb_apply.clicked.connect(self.clickedWhichAre)
        self.pb_combo.clicked.connect(self.tryAllCombo)

    def clickedWhich(self):
        button_name = self.sender()
        print(button_name.objectName())

    def populateChecked(self):
        self.onlyCheckedBoxes = []
        for names in self.all_boxes:
            if names.isChecked():
                self.onlyCheckedBoxes.append(names.objectName())

    QtCore.pyqtSlot()
    def clickedWhichAre(self):
        self.populateChecked()
        self.signal.emit(self.onlyCheckedBoxes)

    QtCore.pyqtSlot()
    def tryAllCombo(self):
        from itertools import combinations
        self.iter_list = list(combinations(self.ref_names[1:],self.sb_max_combo.value()))

        for n, refs in enumerate(self.iter_list):
            self.statusbar.showMessage(f"{n+1}/{len(self.iter_list)}")
            self.signal.emit(list((str(self.ref_names[0]),)+refs))
            QtTest.QTest.qWait(self.sb_time_delay.value()*1000)

    def enableApply(self):
        self.populateChecked()
        if len(self.onlyCheckedBoxes)>1:
            self.pb_apply.setEnabled(True)
        else:
            self.pb_apply.setEnabled(False)


class ScatterPlot(QtWidgets.QMainWindow):

    def __init__(self, img1, img2):
        super(ScatterPlot, self).__init__()

        uic.loadUi('uis/ScatterView.ui', self)
        self.clearPgPlot()
        self.w1 = self.scatterViewer.addPlot()
        self.img1 = img1
        self.img2 = img2

        self.s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))
        '''
        points = []
        
        for i in range(len(self.img1.flatten())):
            x = self.img1.flatten()[i]
            y = self.img2.flatten()[i]
            points.append({'pos': (x,y), 'data': 'id', 'size': 3, 'pen': pg.mkPen(None),
                           'brush': pg.mkBrush(255, 255, 0, 120)})
                           
        self.s1.addPoints(points)
        '''

        self.s1.setData(self.img1.flatten(),self.img2.flatten())
        self.w1.setLabel('bottom','Image ROI')
        self.w1.setLabel('left', 'Math ROI')
        self.w1.addItem(self.s1)

        self.image_view.setImage(self.img1)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('thermal')

        self.image_view2.setImage(self.img2)
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()
        self.image_view2.setPredefinedGradient('thermal')

        # connections
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSave_Images.triggered.connect(self.tiff_export_images)
        self.pb_define_mask.clicked.connect(self.createMask)
        self.pb_apply_mask.clicked.connect(self.getMaskRegion)

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.w1)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save correlation", '', 'spectrum and fit (*csv)')
        exporter.export(str(file_name[0])+'.csv')
        self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, "save images", '', 'spectrum and fit (*tiff)')
        tf.imsave(str(file_name[0]) + '.tiff', np.dstack([self.img1,self.img2]).T)
        self.statusbar.showMessage(f"Images saved to {str(file_name[0])}")

    def createMask(self):

        self.size = self.img1.max()/10
        self.pos = int(self.img1.mean())

        self.scatter_mask = pg.PolyLineROI([[0, 0], [0, self.size], [self.size, self.size], [self.size, 0]],
                                             pos=(self.pos, self.pos), pen='r', closed=True,removable = True)

        self.w1.addItem(self.scatter_mask)

    def resetMask(self):
        self.w1.removeItem(self.scatter_mask)
        self.createMask()

    def clearPgPlot(self):
        try:
            self.masked_img.close()
        except:
            pass


    def getMaskRegion(self):

        # Ref : https://stackoverflow.com/questions/57719303/how-to-map-mouse-position-on-a-scatterplot

        roiShape = self.scatter_mask.mapToItem(self.s1, self.scatter_mask.shape())
        self._points = list()
        for i in range(len(self.img1.flatten())):
            self._points.append(QtCore.QPointF(self.img1.flatten()[i], self.img2.flatten()[i]))


        selected = [roiShape.contains(pt) for pt in self._points]
        img_selected = np.reshape(selected, (self.img1.shape))

        self.clearPgPlot()
        self.masked_img = pg.image()
        self.masked_img.setImage(img_selected * self.img1)
        self.masked_img.setPredefinedGradient('bipolar')
        self.masked_img.setWindowTitle("Masked Image")

















