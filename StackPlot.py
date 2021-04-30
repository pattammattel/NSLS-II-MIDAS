import sys, os, json
import tifffile as tf
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import logging
from itertools import combinations

from PyQt5 import QtWidgets,QtCore,QtGui, uic, QtTest
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph import ImageView, PlotWidget
from PyQt5.QtCore import pyqtSignal

#Custom .py file contains image calculations
from StackCalcs import *

logger = logging.getLogger()


class singleStackViewer(QtWidgets.QMainWindow):
    def __init__(self,  img_stack, gradient = 'viridis'):
        super(singleStackViewer, self).__init__()

        # Load the UI Page
        uic.loadUi('uis/singleStackView.ui', self)


        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.img_stack = img_stack
        self.gradient = gradient
        self.image_view.setPredefinedGradient(gradient)

        if self.img_stack.ndim == 3:
            self.dim1, self.dim3, self.dim2 = img_stack.shape
        elif self.img_stack.ndim == 2:
            self.dim3, self.dim2 = img_stack.shape
            self.dim1 = 1
        self.hs_img_stack.setMaximum(self.dim1-1)
        self.hs_img_stack.setValue(np.round(self.dim1/2))
        self.displayStack()

        #connections
        self.hs_img_stack.valueChanged.connect(self.displayStack)
        self.actionSave.triggered.connect(self.saveImageStackAsTIFF)

    def displayStack(self):
        im_index = self.hs_img_stack.value()
        if self.img_stack.ndim == 2:
            self.image_view.setImage(self.img_stack)
        else:
            self.image_view.setImage(self.img_stack[im_index])
        self.label_img_count.setText(f'{im_index+1}/{self.dim1}')

    def saveImageStackAsTIFF(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', '*.tiff;;*.tif')
        if file_name[0]:
            if self.img_stack.ndim == 3:
                tf.imsave(str(file_name[0]), np.float32(self.img_stack.transpose(0, 2, 1)))
            elif self.img_stack.ndim == 2:
                tf.imsave(str(file_name[0]), np.float32(self.img_stack.T))
        else:
            pass


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
        self.label_comp_number.setText(f'{im_index+1}/{self.dim1}')
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])

    def show_all_spec(self):
        self.spectrum_view.clear()
        self.plt_colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w'] * 3
        offsets = np.arange(0, 2, 0.2)
        self.spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.spectrum_view.plot(self.energy,(self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                                    pen=self.plt_colors[ii], name="component" + str(ii + 1))

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '_components.tiff', np.float32(self.comp_stack.transpose(0, 2, 1)), imagej=True)
            tf.imsave(str(file_name[0]) + '_component_masks.tiff', np.float32(self.decomp_map.T),imagej=True)
            np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)
            np.savetxt(str(file_name[0]) + '_component_spec.txt', self.comp_spectra)
        else:
            pass

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
        self.label_comp_number.setText(f'{im_index + 1}/{self.dim1}')

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

    def __init__(self, im_stack=None, e_list = None, refs = None, ref_names = None):
        super(XANESViewer, self).__init__()

        uic.loadUi('uis/XANESViewer.ui', self)
        self.centralwidget.setStyleSheet(open('defaultStyle.css').read())

        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs
        self.ref_names = ref_names
        self.selected = self.ref_names
        self.fitResultDict = {}
        self.fit_method = self.cb_xanes_fit_model.currentText()

        self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack, self.e_list,
                                                                      self.refs, method=self.fit_method)

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
        self.pb_re_fit.clicked.connect(self.re_fit_xanes)
        self.pb_edit_refs.clicked.connect(self.choose_refs)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.pb_save_chem_map.clicked.connect(self.save_chem_map)
        self.pb_save_spe_fit.clicked.connect(self.pg_export_spec_fit)
        self.hsb_xanes_stk.valueChanged.connect(self.display_image_data)
        self.hsb_chem_map.valueChanged.connect(self.display_image_data)
        self.actionexportResults.triggered.connect(self.exportFitResults)
        #self.actionexportResults.triggered.connect(self.exportFitResults)

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
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=pg.mkPen(self.plt_colors[ii],width=2),
                                             name=self.selected[1:][ii])
            else:
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=pg.mkPen(self.plt_colors[ii],width=2),
                                             name="ref" + str(ii + 1))

    def choose_refs(self):
        'Interactively exclude some standards from the reference file'
        self.ref_edit_window = RefChooser(self.ref_names,self.im_stack,self.e_list,
                                          self.refs, self.sb_e_shift.value(),
                                          self.cb_xanes_fit_model.currentText())
        self.ref_edit_window.show()
        #self.rf_plot = pg.plot(title="RFactor Tracker")

        #connections
        self.ref_edit_window.choosenRefsSignal.connect(self.update_refs)
        self.ref_edit_window.fitResultsSignal.connect(self.plotFitResults)

    def update_refs(self,list_):
        self.selected = list_ # list_ is the signal from ref chooser
        self.update_spectrum()
        self.re_fit_xanes()

    def update_spectrum(self):

        self.roi_img = self.image_roi.getArrayRegion(self.im_stack, self.image_view.imageItem, axes=(1, 2))
        sizex, sizey = self.roi_img.shape[1], self.roi_img.shape[2]
        posx, posy = self.image_roi.pos()
        self.roi_info.setText(f'ROI_Pos: {int(posx)},{int(posy)} ROI_Size: {sizex},{sizey}')

        self.xdata1 = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_sum_spectra(self.roi_img)
        self.fit_method = self.cb_xanes_fit_model.currentText()

        if len(self.selected) != 0:

            self.inter_ref = interploate_E(self.refs[self.selected], self.xdata1)
            stats, coeffs = xanes_fitting_1D(self.ydata1, self.xdata1, self.refs[self.selected],
                                             method=self.fit_method, alphaForLM=0.05)

        else:
            self.inter_ref = interploate_E(self.refs, self.xdata1)
            stats, coeffs = xanes_fitting_1D(self.ydata1, self.xdata1, self.refs,
                                             method=self.fit_method, alphaForLM=0.05)

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
        #set the rfactor value to the line edit slot
        self.results = f"Coefficients: {coeffs} \n"\
                       f"R-Factor: {stats['R_Factor']}, R-Square: {stats['R_Square']},\n "\
                       f"Chi-Square: {stats['Chi_Square']}, "\
                       f"Reduced Chi-Square: {stats['Reduced Chi_Square']}"

        self.fit_results.setText(self.results)

    def re_fit_xanes(self):
        if len(self.selected) != 0:
            self.decon_ims, self.rfactor, self.coeffs_arr  = xanes_fitting(self.im_stack, self.e_list + self.sb_e_shift.value(),
                                       self.refs[self.selected], method=self.cb_xanes_fit_model.currentText())
        else:
            #if non athena file with no header is loaded no ref file cannot be edited
            self.decon_ims,self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack, self.e_list + self.sb_e_shift.value(),
                                       self.refs, method=self.cb_xanes_fit_model.currentText())

        #rfactor is a list of all spectra so take the mean
        self.rfactor_mean = np.mean(self.rfactor)
        self.image_view_maps.setImage(self.decon_ims.transpose(2,0,1))
        self.scrollBar_setup()

    def plotFitResults(self,decon_ims,rfactor_mean,coeff_array):
        #upadte the chem maps and scrollbar params
        self.image_view_maps.setImage(decon_ims.transpose(2, 0, 1))
        #self.hsb_chem_map.setValue(0)
        #self.hsb_chem_map.setMaximum(decon_ims.shape[-1]-1)

        #set the rfactor value to the line edit slot
        self.le_r_sq.setText(f'{rfactor_mean :.4f}')

    def save_chem_map(self):
        file_name = QFileDialog().getSaveFileName(self, "save image", '', 'image data (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '_xanes_map.tiff', np.float32(self.decon_ims.T), imagej=True)
            tf.imsave(str(file_name[0]) + '_rfactor.tiff', np.float32(self.rfactor.T), imagej=True)
        else:
            logger.error('No file to save')
            pass

    def save_spec_fit(self):
        try:
            to_save = np.column_stack([self.xdata1, self.ydata1, self.fit_])
            file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectrum and fit (*txt)')
            if file_name[0]:
                np.savetxt(str(file_name[0]) + '.txt', to_save)
            else:
                pass
        except:
            logger.error('No file to save')
            pass

    def pg_export_spec_fit(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectrum and fit (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0])+'.csv')
        else:
            pass

    def exportFitResults(self):
        file_name = QFileDialog().getSaveFileName(self, "save txt", 'xanes_1D_fit_results.txt', 'txt data (*txt)')
        if file_name[0]:
            with open(file_name[0], 'w') as file:
                file.write(self.results)
        else:
            pass


class RefChooser(QtWidgets.QMainWindow):
    choosenRefsSignal: pyqtSignal = QtCore.pyqtSignal(list)
    fitResultsSignal: pyqtSignal = QtCore.pyqtSignal(np.ndarray, float, np.ndarray)

    def __init__(self, ref_names,im_stack,e_list, refs, e_shift, fit_model):
        super(RefChooser, self).__init__()
        uic.loadUi('uis/RefChooser.ui', self)
        self.ref_names = ref_names
        self.refs = refs
        self.im_stack = im_stack
        self.e_list = e_list
        self.e_shift = e_shift
        self.fit_model = fit_model

        self.all_boxes = []
        self.rFactorList = []

        self.displayCombinations()

        # selection become more apparent than default with red-ish color
        self.tableWidget.setStyleSheet("background-color: white; selection-background-color: rgb(200,0,0);")

        # add a line to the plot to walk through the table. Note that the table is not sorted
        self.selectionLine = pg.InfiniteLine(pos=1, angle=90, pen=pg.mkPen('m', width=2.5),
                                             movable=True, bounds=None, label='Move Me!')
        self.stat_view.setLabel('bottom', 'Fit ID')
        self.stat_view.setLabel('left', 'R-Factor')

        for n, i in enumerate(self.ref_names):
            self.cb_i = QtWidgets.QCheckBox(self.ref_box_frame)
            if n == 0:
                self.cb_i.setChecked(True)
                self.cb_i.setEnabled(False)
            self.cb_i.setObjectName(i)
            self.cb_i.setText(i)
            self.gridLayout_2.addWidget(self.cb_i, n, 0, 1, 1)
            self.cb_i.toggled.connect(self.enableApply)
            self.all_boxes.append(self.cb_i)

        #connections
        self.pb_apply.clicked.connect(self.clickedWhichAre)
        self.pb_combo.clicked.connect(self.tryAllCombo)
        self.actionExport_Results_csv.triggered.connect(self.exportFitResults)
        self.selectionLine.sigPositionChanged.connect(self.updateFitWithLine)
        self.tableWidget.itemSelectionChanged.connect(self.updateWithTableSelection)
        #self.stat_view.scene().sigMouseClicked.connect(self.moveSelectionLine)
        self.stat_view.mouseDoubleClickEvent  = self.moveSelectionLine
        self.sb_max_combo.valueChanged.connect(self.displayCombinations)
        #self.pb_sort_with_r.clicked.connect(lambda: self.tableWidget.sortItems(3, QtCore.Qt.AscendingOrder))
        self.pb_sort_with_r.clicked.connect(self.sortTable)
        self.cb_sorter.currentTextChanged.connect(self.sortTable)

    def clickedWhich(self):
        button_name = self.sender()

    def populateChecked(self):
        self.onlyCheckedBoxes = []
        for names in self.all_boxes:
            if names.isChecked():
                self.onlyCheckedBoxes.append(names.objectName())

    QtCore.pyqtSlot()
    def clickedWhichAre(self):
        self.populateChecked()
        self.choosenRefsSignal.emit(self.onlyCheckedBoxes)

    def generateRefList(self, ref_list , maxCombo):
        iter_list = []
        i = 1
        while i<maxCombo+1:
            iter_list += list(combinations(ref_list,i))
            i += 1
        return len(iter_list), iter_list

    def displayCombinations(self):
        niter, self.iter_list = self.generateRefList(self.ref_names[1:], self.sb_max_combo.value())
        self.label_nComb.setText(str(niter) + " Combinations")

    @QtCore.pyqtSlot()
    def tryAllCombo(self):
        self.rfactor_list = []
        self.df = pd.DataFrame(columns=['Fit Number','References', 'Coefficients',
                                        'R-Factor', 'R^2', 'chi^2', 'red-chi^2'])

        self.tableWidget.setHorizontalHeaderLabels(self.df.columns)
        #self.iter_list = list(combinations(self.ref_names[1:],self.sb_max_combo.value()))
        niter, self.iter_list = self.generateRefList(self.ref_names[1:],self.sb_max_combo.value())
        tot_combo = len(self.iter_list)
        for n, refs in enumerate(self.iter_list):
            self.statusbar.showMessage(f"{n+1}/{tot_combo}")
            selectedRefs = (list((str(self.ref_names[0]),)+refs))
            self.fit_combo_progress.setValue((n + 1) * 100 / tot_combo)
            self.stat, self.coeffs_arr  = xanes_fitting_Line(self.im_stack, self.e_list + self.e_shift,
                                                                           self.refs[selectedRefs], method=self.fit_model)

            self.rfactor_list.append(self.stat['R_Factor'])
            self.stat_view.plot(x = np.arange(n+1),y = self.rfactor_list, clear = True,title = 'R-Factor',
                                pen = pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')

            resultsDict = {'Fit Number':n,'References':str(selectedRefs[1:]),
                           'Coefficients': str(np.around(self.coeffs_arr,4)),
                           'R-Factor':self.stat['R_Factor'],'R^2':self.stat['R_Square'],
                           'chi^2':self.stat['Chi_Square'], 'red-chi^2':self.stat['Reduced Chi_Square'] }

            df2 = pd.DataFrame([resultsDict])
            self.df = pd.concat([self.df,df2],ignore_index=True)

            self.dataFrametoQTable(self.df)
            QtTest.QTest.qWait(0.1) # hepls with real time plotting

        self.stat_view.addItem(self.selectionLine)

    def dataFrametoQTable(self, df_:pd.DataFrame):
        nRows = len(df_.index)
        nColumns = len(df_.columns)
        self.tableWidget.setRowCount(nRows)
        self.tableWidget.setColumnCount(nColumns)
        self.tableWidget.setHorizontalHeaderLabels(df_.columns)

        for i in range(nRows):
            for j in range(nColumns):
                cell = QtWidgets.QTableWidgetItem(str(df_.values[i][j]))
                self.tableWidget.setItem(i, j, cell)

        # set the property of the table view. Size policy to make the contents justified
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget.resizeColumnsToContents()

    def exportFitResults(self):
        file_name = QFileDialog().getSaveFileName(self, "save csv", 'xanes_fit_results_log.csv', 'txt data (*csv)')
        if file_name[0]:
            with open(str(file_name[0]), 'w') as fp:
                self.df.to_csv(fp)
        else:
            pass

    def selectTableAndCheckBox(self,x):
        nSelection = int(round(x))
        self.tableWidget.selectRow(nSelection)
        fit_num = int(self.tableWidget.item(nSelection, 0).text())
        refs_selected = self.iter_list[fit_num]

        #reset all the checkboxes to uncheck state, except the energy
        for checkstate in self.findChildren(QtWidgets.QCheckBox):
            if checkstate.isEnabled():
                checkstate.setChecked(False)

        for cb_names in refs_selected:
            checkbox = self.findChild(QtWidgets.QCheckBox, name = cb_names)
            checkbox.setChecked(True)

    def updateFitWithLine(self):
        pos_x,pos_y = self.selectionLine.pos()
        x = self.df.index[self.df[str('Fit Number')] == np.round(pos_x)][0]
        self.selectTableAndCheckBox(x)

    def updateWithTableSelection(self):
        x = self.tableWidget.currentRow()
        self.selectTableAndCheckBox(x)

    def moveSelectionLine(self,event):
        if event.button() == QtCore.Qt.LeftButton:
            Pos = self.stat_view.plotItem.vb.mapSceneToView(event.pos())
            self.selectionLine.setPos(Pos.x())

    def sortTable(self):
        sorter_dict = {'R-Factor':'R-Factor','R-Square':'R^2',
                       'Chi-Square':'chi^2', 'Reduced Chi-Square':'red-chi^2',
                       'Fit Number':'Fit Number'}
        sorter = sorter_dict[self.cb_sorter.currentText()]
        self.df = self.df.sort_values(sorter, ignore_index=True)
        self.dataFrametoQTable(self.df)

    def enableApply(self):

        """  """
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
        if file_name[0]:
            exporter.export(str(file_name[0])+'.csv')
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            pass

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, "save images", '', 'spectrum and fit (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '.tiff', np.dstack([self.img1,self.img2]).T)
            self.statusbar.showMessage(f"Images saved to {str(file_name[0])}")
        else:
            pass

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

        self.masked_img = singleStackViewer(img_selected * self.img1, gradient='bipolar')
        self.masked_img.show()

class LoadingScreen(QtWidgets.QSplashScreen):
    def __init__(self):
        super(LoadingScreen, self).__init__()
        uic.loadUi('uis/animationWindow.ui', self)
        self.setWindowOpacity(0.65)
        self.movie = QMovie("uis/animation.gif")
        self.label.setMovie(self.movie)

    def mousePressEvent(self, event):
        # disable default "click-to-dismiss" behaviour
        pass

    def startAnimation(self):
        self.movie.start()
        self.show()

    def stopAnimation(self):
        self.movie.stop()
        self.hide()


















