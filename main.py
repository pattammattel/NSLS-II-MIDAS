# -*- coding: utf-8 -*-

# Author: Ajith Pattammattel
# First Version on:06-23-2020

import logging, sys, webbrowser, traceback, os, json,h5py
import scipy.stats as stats
import numpy as np
import pandas as pd
import tifffile as tf
import pyqtgraph as pg
import pyqtgraph.exporters
import scipy.optimize as opt
import sklearn.decomposition as sd
import sklearn.cluster as sc
import larch

from pyqtgraph import plot,ImageView, PlotWidget
from itertools import combinations
from scipy.stats import linregress
from scipy.signal import savgol_filter
from skimage.transform import resize
from skimage import filters
from sklearn import linear_model
from larch.xafs import preedge
from pystackreg import StackReg

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5 import QtWidgets, QtCore, QtGui, uic, QtTest
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDesktopWidget, QApplication, QSizePolicy
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool

from MultiChannel import *

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))
pg.setConfigOption('imageAxisOrder', 'row-major')


if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class midasWindow(QtWidgets.QMainWindow):

    def __init__(self, im_stack=None, energy=[], refs=[]):
        super(midasWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path, 'uis/midasMainwindow.ui'), self)
        self.im_stack = im_stack
        self.energy = energy
        self.refs = refs
        self.loaded_tranform_file = []
        self.image_roi2_flag = False
        self.refStackAvailable = False
        self.reloadStack = False
        self.plotWidth = 2

        self.plt_colors = ['g', 'r', 'c', 'm', 'y', 'w', 'b',
                           pg.mkPen(70, 5, 80), pg.mkPen(255, 85, 130),
                           pg.mkPen(0, 85, 130), pg.mkPen(255, 170, 60)] * 3
        # window style
        self.actionDarkMode.triggered.connect(self.darkMode)
        self.actionDefault.triggered.connect(self.defaultMode)
        self.actionModern.triggered.connect(self.modernMode)

        #self.setToolTipsVisible(True)
        for menuItem in self.findChildren(QtWidgets.QMenu):
            menuItem.setToolTipsVisible(True)

        # plotview options
        self.actionWhite.triggered.connect(lambda: self.spectrum_view.setBackground('w'))
        self.actionRed.triggered.connect(lambda: self.spectrum_view.setBackground('r'))
        self.actionYellow.triggered.connect(lambda: self.spectrum_view.setBackground('y'))
        self.actionBlue.triggered.connect(lambda: self.spectrum_view.setBackground('b'))
        self.actionBlack.triggered.connect(lambda: self.spectrum_view.setBackground((0, 0, 0)))

        self.actn1.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn1.text())))
        self.actn2.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn2.text())))
        self.actn3.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn3.text())))
        self.actn4.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn4.text())))
        self.actn5.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn5.text())))
        self.actn6.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn6.text())))
        self.actn8.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn8.text())))
        self.actn10.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn10.text())))

        self.actionOpen_Image_Data.triggered.connect(self.browse_file)
        self.actionOpen_Multiple_Files.triggered.connect(self.createVirtualStack)
        self.actionSave_as.triggered.connect(self.save_stack)
        self.actionExit.triggered.connect(lambda: QApplication.closeAllWindows())
        self.actionOpen_in_GitHub.triggered.connect(self.open_github_link)
        self.actionLoad_Energy.triggered.connect(self.select_elist)
        self.menuFile.setToolTipsVisible(True)


        #Accessories
        self.actionOpen_Mask_Gen.triggered.connect(self.openMaskMaker)
        self.actionMultiColor.triggered.connect(self.openMultiColorWindow)

        #calculations
        self.pb_transpose_stack.clicked.connect(lambda: self.threadMaker(self.transposeStack))
        self.pb_swapXY_stack.clicked.connect(lambda: self.threadMaker(self.swapStackXY))
        self.pb_reset_img.clicked.connect(self.reloadImageStack)
        self.pb_crop.clicked.connect(self.crop_to_dim)
        self.pb_crop.clicked.connect(self.view_stack)
        self.sb_scaling_factor.valueChanged.connect(self.view_stack)
        self.pb_ref_xanes.clicked.connect(self.select_ref_file)
        self.pb_elist_xanes.clicked.connect(self.select_elist)

        [uis.valueChanged.connect(self.replot_image) for uis in
         [self.hs_smooth_size, self.hs_nsigma, self.hs_bg_threshold]]

        [uis.stateChanged.connect(self.replot_image) for uis in
         [self.cb_remove_bg, self.cb_remove_outliers, self.cb_smooth,
          self.cb_norm, self.cb_log]]

        [uis.stateChanged.connect(self.view_stack) for uis in
         [self.cb_remove_edges, self.cb_upscale,
          self.cb_rebin]]

        # ToolBar
        self.actionStack_Info.triggered.connect(self.displayStackInfo)
        self.actionSave_Image.triggered.connect(self.save_disp_img)
        self.actionExport_Stack.triggered.connect(self.save_stack)

        # ROI background
        self.actionSubtract_ROI_BG.triggered.connect(lambda: self.threadMaker(self.removeROIBGStack))

        # alignment
        self.pb_load_align_ref.clicked.connect(self.loadAlignRefImage)
        self.pb_loadAlignTranform.clicked.connect(self.importAlignTransformation)
        self.pb_saveAlignTranform.clicked.connect(self.exportAlignTransformation)
        self.pb_alignStack.clicked.connect(lambda: self.threadMaker(self.stackRegistration))
        # self.pb_alignStack.clicked.connect(self.stackRegistration)

        # save_options
        self.actionSave_Sum_Image.triggered.connect(lambda: self.save_stack(saveSum=True))
        self.pb_save_disp_spec.clicked.connect(self.save_disp_spec)
        self.actionSave_Energy_List.triggered.connect(self.saveEnergyList)
        self.pb_show_roi.clicked.connect(self.getROIMask)
        self.pb_addToCollector.clicked.connect(self.addSpectrumToCollector)
        self.pb_collect_clear.clicked.connect(lambda: self.spectrum_view_collect.clear())
        self.pb_saveCollectorPlot.clicked.connect(self.saveCollectorPlot)

        # XANES Normalization
        self.pb_apply_xanes_norm.clicked.connect(self.nomalizeLiveSpec)
        self.pb_auto_Eo.clicked.connect(self.findEo)
        self.pb_xanes_norm_vals.clicked.connect(self.initNormVals)
        self.pb_apply_norm_to_stack.clicked.connect(lambda: self.threadMaker(self.normalizeStack))
        self.actionExport_Norm_Params.triggered.connect(self.exportNormParams)
        self.actionImport_Norm_Params.triggered.connect(self.importNormParams)

        # Analysis
        self.pb_pca_scree.clicked.connect(self.pca_scree_)
        self.pb_calc_components.clicked.connect(self.calc_comp_)
        self.pb_kmeans_elbow.clicked.connect(self.kmeans_elbow)
        self.pb_calc_cluster.clicked.connect(self.clustering_)
        self.pb_xanes_fit.clicked.connect(self.fast_xanes_fitting)
        self.pb_plot_refs.clicked.connect(self.plt_xanes_refs)

        self.show()

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum  {self.threadpool.maxThreadCount()} threads")

    # View Options
    def darkMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path, 'css/darkStyle.css')).read())

    def defaultMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path, 'css/defaultStyle.css')).read())

    def modernMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path, 'css/modern.css')).read())

    def setPlotLineWidth(self, width_input):
        self.plotWidth = width_input
        try:
            self.update_spectrum()
        except:
            pass

    def openMultiColorWindow(self):
        self.multicolorwindow = MultiChannelWindow()
        self.multicolorwindow.show()

    def openMaskMaker(self):
        self.mask_window = MaskSpecViewer(xanes_stack=self.displayedStack, energy=self.energy)
        self.mask_window.show()

    def open_github_link(self):
        webbrowser.open('https://github.com/pattammattel/NSLS-II-MIDAS/wiki')

    def threadMaker(self, funct):
        # Pass the function to execute
        worker = Worker(funct)  # Any other args, kwargs are passed to the run function
        self.loadSplashScreen()
        worker.signals.start.connect(self.splash.startAnimation)
        worker.signals.result.connect(self.print_output)

        list(map(worker.signals.finished.connect, [self.thread_complete, self.splash.stopAnimation,
                                                   self.update_stack_info,self.update_spectrum,
                                                   self.update_image_roi]))

        # Execute
        self.threadpool.start(worker)

    # File Loading
    def createVirtualStack(self):
        """ User can load multiple/series of tiff images with same shape.
        The 'self.load_stack()' recognizes 'self.filename as list and create the stack.
        """
        self.energy = []
        filter = "TIFF (*.tiff);;TIF (*.tif);;all_files (*)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files", " ", filter)
        if names[0]:

            self.file_name = names[0]
            self.load_stack()

        else:
            self.statusbar_main.showMessage("No file has selected")
            pass

    def load_stack(self):

        """ load the image data from the selected file.
        If the the choice is for multiple files stack will be created in a loop.
        If single h5 file is selected the unpacking will be done with 'get_xrf_data' function in StackCalcs.
        From the h5 the program can recognize the beamline. The exported stack will be normalized to I0.

        If the single tiff file is choosen tf.imread() is used.

        The output 'self.im_stack' is the unmodified data file
        """

        self.log_warning = False  # for the Qmessage box in cb_log
        self.image_roi2_flag = False
        self.cb_log.setChecked(False)
        self.cb_remove_edges.setChecked(False)
        self.cb_norm.setChecked(False)
        self.cb_smooth.setChecked(False)
        self.cb_remove_outliers.setChecked(False)
        self.cb_remove_bg.setChecked(False)
        self.cb_rebin.setChecked(False)
        self.cb_upscale.setChecked(False)
        self.sb_xrange1.setValue(0)
        self.sb_yrange1.setValue(0)
        self.sb_zrange1.setValue(0)

        self.menuMask.setEnabled(True)
        self.actionLoad_Energy.setEnabled(True)
        self.actionSave_Energy_List.setEnabled(True)
        self.actionSave_as.setEnabled(True)

        self.sb_zrange2.setMaximum(99999)
        self.sb_xrange2.setMaximum(99999)
        self.sb_yrange2.setMaximum(99999)

        self.statusbar_main.showMessage('Loading.. please wait...')

        if isinstance(self.file_name, list):

            all_images = []

            for im_file in self.file_name:
                img = tf.imread(im_file)
                all_images.append(img) #row major image
            self.im_stack = np.dstack(all_images).transpose((2,0,1))
            self.avgIo = 1  # I0 is only applicable to XRF h5 files
            self.sb_zrange2.setValue(self.im_stack.shape[0])

        else:

            if self.file_name.endswith('.h5'):
                self.im_stack, mono_e, bl_name, self.avgIo = get_xrf_data(self.file_name)
                self.statusbar_main.showMessage(f'Data from {bl_name}')
                self.sb_zrange2.setValue(mono_e / 10)
                self.energy = []

            elif self.file_name.endswith('.tiff') or self.file_name.endswith('.tif'):
                self.im_stack_ = tf.imread(self.file_name)
                if self.im_stack_.ndim == 2:
                    self.im_stack = self.im_stack_.reshape(1, self.im_stack_.shape[0], self.im_stack_.shape[1])

                else:
                    self.im_stack = self.im_stack_
                self.sb_zrange2.setValue(self.im_stack.shape[0])
                self.autoEnergyLoader()
                self.energyUnitCheck()
                self.avgIo = 1

            else:
                logger.error('Unknown data format')

        """ Fill the stack dimensions to the GUI and set the image dimensions as max values.
         This prevent user from choosing higher image dimensions during a resizing event"""

        logger.info(f' loaded stack with {np.shape(self.im_stack)} from the file')

        try:
            logger.info(f' Transposed to shape: {np.shape(self.im_stack)}')
            self.init_dimZ, self.init_dimY, self.init_dimX = self.im_stack.shape
            # Remove any previously set max value during a reload

            self.sb_xrange2.setValue(self.init_dimX)
            self.sb_yrange2.setValue(self.init_dimY)

        except UnboundLocalError:
            logger.error('No file selected')
            pass

        self.view_stack()
        logger.info("Stack displayed correctly")
        self.update_stack_info()

        logger.info(f'completed image shape {np.shape(self.im_stack)}')

        try:
            self.statusbar_main.showMessage(f'Loaded: {self.file_name}')

        except AttributeError:
            self.statusbar_main.showMessage('New Stack is made from selected tiffs')
            pass

    def browse_file(self):
        """ To open a file widow and choose the data file.
        The filename will be used to load data using 'rest and load stack' function """

        filename = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*.hdf *.h5 *tiff *tif )')
        self.file_name = (str(filename[0]))

        # if user decides to cancel the file window gui returns to original state
        if self.file_name:
            self.load_stack()

        else:
            self.statusbar_main.showMessage("No file has selected")
            pass

    def autoEnergyLoader(self):

        dir_, filename_ = os.path.split(self.file_name)
        self.efilePath_name = os.path.join(dir_, os.path.splitext(filename_)[0] + '.txt')
        self.efilePath_log = os.path.join(dir_, 'maps_log_tiff.txt')

        if os.path.isfile(self.efilePath_name):
            self.efilePath = self.efilePath_name
            self.efileLoader()
            self.statusbar_main.showMessage(f"Energy File detected {self.efilePath}")

        elif os.path.isfile(self.efilePath_log):
            self.efilePath = self.efilePath_log
            self.efileLoader()
            self.statusbar_main.showMessage(f"Energy File detected {self.efilePath}")

        else:
            self.efilePath = False
            self.efileLoader()


    def update_stack_info(self):
        z, y, x = np.shape(self.displayedStack)
        self.sb_zrange2.setMaximum(z + self.sb_zrange1.value())
        self.sb_xrange2.setValue(x)
        self.sb_xrange2.setMaximum(x)
        self.sb_yrange2.setValue(y)
        self.sb_yrange2.setMaximum(y)
        logger.info('Stack info has been updated')

    # Image Transformations

    def crop_to_dim(self):
        self.x1, self.x2 = self.sb_xrange1.value(), self.sb_xrange2.value()
        self.y1, self.y2 = self.sb_yrange1.value(), self.sb_yrange2.value()
        self.z1, self.z2 = self.sb_zrange1.value(), self.sb_zrange2.value()

        try:
            self.displayedStack = remove_nan_inf(self.displayedStack[self.z1:self.z2,
                                                self.y1:self.y2, self.x1:self.x2])
        except:
            self.displayedStack = remove_nan_inf(self.im_stack[self.z1:self.z2,
                                                self.y1:self.y2, self.x1:self.x2])

    def transpose_stack(self):
        self.displayedStack = self.displayedStack.T
        self.update_spectrum()
        self.update_spec_image_roi()

    # Alignement

    def loadAlignRefImage(self):
        filename = QFileDialog().getOpenFileName(self, "Image Data", '', '*.tiff *.tif')
        file_name = (str(filename[0]))
        self.alignRefImage = tf.imread(file_name)
        assert self.alignRefImage.shape == self.displayedStack.shape, "Image dimensions do not match"
        self.refStackAvailable = True
        self.rb_alignRefVoid.setChecked(False)
        self.change_color_on_load(self.pb_load_align_ref)

    def stackRegistration(self):

        self.transformations = {
            'TRANSLATION': StackReg.TRANSLATION,
            'RIGID_BODY': StackReg.RIGID_BODY,
            'SCALED_ROTATION': StackReg.SCALED_ROTATION,
            'AFFINE': StackReg.AFFINE,
            'BILINEAR': StackReg.BILINEAR
        }

        self.transformType = self.transformations[self.cb_alignTransform.currentText()]
        self.alignReferenceImage = self.cb_alignRef.currentText()
        self.alignRefStackVoid = self.rb_alignRefVoid.isChecked()
        self.alignMaxIter = self.sb_maxIterVal.value()

        if self.cb_use_tmatFile.isChecked():

            if len(self.loaded_tranform_file) > 0:

                self.displayedStack = align_with_tmat(self.displayedStack, tmat_file=self.loaded_tranform_file,
                                                     transformation=self.transformType)
                logger.info("Aligned to the tranform File")

            else:
                logger.error("No Tranformation File Loaded")


        elif self.cb_iterAlign.isChecked():

            if not self.refStackAvailable:
                self.alignRefImage = self.displayedStack
            else:
                pass

            self.displayedStack = align_stack_iter(self.displayedStack, ref_stack_void=False,
                                                  ref_stack=self.alignRefImage, transformation=self.transformType,
                                                  method=('previous', 'first'), max_iter=self.alignMaxIter)

        else:
            if not self.refStackAvailable:
                self.alignRefImage = self.displayedStack

            else:
                pass

            self.displayedStack, self.tranform_file = align_stack(self.displayedStack, ref_image_void=True,
                                                                 ref_stack=self.alignRefImage,
                                                                 transformation=self.transformType,
                                                                 reference=self.alignReferenceImage)
            logger.info("New Tranformation file available")
        self.im_stack = self.displayedStack

    def exportAlignTransformation(self):
        file_name = QFileDialog().getSaveFileName(self, "Save Transformation File", 'TranformationMatrix.npy',
                                                  'text file (*.npy)')
        if file_name[0]:
            np.save(file_name[0], self.tranform_file)
        else:
            pass

    def importAlignTransformation(self):
        file_name = QFileDialog().getOpenFileName(self, "Open Transformation File", ' ',
                                                  'text file (*.npy)')
        if file_name[0]:
            self.loaded_tranform_file = np.load(file_name[0])
            self.cb_use_tmatFile.setChecked(True)
            logger.info("Tranformation File Loaded")
        else:
            pass

    def loadSplashScreen(self):
        self.splash = LoadingScreen()

        px = self.geometry().x()
        py = self.geometry().y()
        ph = self.geometry().height()
        pw = self.geometry().width()
        dw = self.splash.width()
        dh = self.splash.height()
        new_x, new_y = px + (0.5 * pw) - dw, py + (0.5 * ph) - dh
        self.splash.setGeometry(new_x, new_y, dw, dh)

        self.splash.show()

    def reloadImageStack(self):

        self.reloadStack = True
        self.load_stack()
        self.reloadStack = False

    def update_stack(self):
        self.displayedStack = self.im_stack
        self.crop_to_dim()

        if self.cb_rebin.isChecked():
            self.cb_upscale.setChecked(False)
            self.sb_scaling_factor.setEnabled(True)
            self.displayedStack = resize_stack(self.displayedStack,
                                              scaling_factor=self.sb_scaling_factor.value())
            self.update_stack_info()

        elif self.cb_upscale.isChecked():
            self.cb_rebin.setChecked(False)
            self.sb_scaling_factor.setEnabled(True)
            self.displayedStack = resize_stack(self.displayedStack, upscaling=True,
                                              scaling_factor=self.sb_scaling_factor.value())
            self.update_stack_info()

        if self.cb_remove_outliers.isChecked():
            self.hs_nsigma.setEnabled(True)
            nsigma = self.hs_nsigma.value() / 10
            self.displayedStack = remove_hot_pixels(self.displayedStack,
                                                   NSigma=nsigma)
            self.label_nsigma.setText(str(nsigma))
            logger.info(f'Removing Outliers with NSigma {nsigma}')

        elif self.cb_remove_outliers.isChecked() == False:
            self.hs_nsigma.setEnabled(False)

        if self.cb_remove_edges.isChecked():
            self.displayedStack = remove_edges(self.displayedStack)
            logger.info(f'Removed edges, new shape {self.displayedStack.shape}')
            self.update_stack_info()

        if self.cb_remove_bg.isChecked():
            self.hs_bg_threshold.setEnabled(True)
            logger.info('Removing background')
            bg_threshold = self.hs_bg_threshold.value()
            self.label_bg_threshold.setText(str(bg_threshold) + '%')
            self.displayedStack = clean_stack(self.displayedStack,
                                             auto_bg=False,
                                             bg_percentage=bg_threshold)

        elif self.cb_remove_bg.isChecked() == False:
            self.hs_bg_threshold.setEnabled(False)

        if self.cb_log.isChecked():

            self.displayedStack = remove_nan_inf(np.log10(self.displayedStack))
            logger.info('Log Stack is in use')

        if self.cb_smooth.isChecked():
            self.hs_smooth_size.setEnabled(True)
            window = self.hs_smooth_size.value()
            if window % 2 == 0:
                window = +1
            self.smooth_winow_size.setText('Window size: ' + str(window))
            self.displayedStack = smoothen(self.displayedStack, w_size=window)
            logger.info('Spectrum Smoothening Applied')

        elif self.cb_smooth.isChecked() == False:
            self.hs_smooth_size.setEnabled(False)

        if self.cb_norm.isChecked():
            logger.info('Normalizing spectra')
            self.displayedStack = normalize(self.displayedStack,
                                           norm_point=-1)

        logger.info(f'Updated image is in use')

    # ImageView

    def view_stack(self):

        if not self.im_stack.ndim == 3:
            raise ValueError("stack should be an ndarray with ndim == 3")
        else:
            self.update_stack()
            # self.StackUpdateThread()

        try:
            self.image_view.removeItem(self.image_roi_math)
        except:
            pass

        (self.dim1, self.dim2, self.dim3) = self.displayedStack.shape
        self.image_view.setImage(self.displayedStack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')
        self.image_view.setCurrentIndex(self.dim1 // 2)
        if len(self.energy) == 0:
            self.energy = np.arange(self.z1, self.z2) * 10
            logger.info("Arbitary X-axis used in the plot for XANES")
        self.sz = np.max(
            [int(self.dim2 * 0.1), int(self.dim3 * 0.1)])  # size of the roi set to be 10% of the image area

        self.stack_center = (self.energy[len(self.energy) // 2])
        self.stack_width = (self.energy.max() - self.energy.min()) // 10
        self.spec_roi = pg.LinearRegionItem(values=(self.stack_center - self.stack_width,
                                                    self.stack_center + self.stack_width))

        # a second optional ROI for calculations follow
        self.image_roi_math = pg.PolyLineROI([[0, 0], [0, self.sz], [self.sz, self.sz], [self.sz, 0]],
                                             pos=(int(self.dim3 // 3), int(self.dim2 // 3)),
                                             pen='r', closed=True, removable=True)

        self.spec_roi_math = pg.LinearRegionItem(values=(self.stack_center - self.stack_width - 10,
                                                         self.stack_center + self.stack_width - 10), pen='r',
                                                 brush=QtGui.QColor(0, 255, 200, 50)
                                                 )
        self.spec_lo_m_idx = self.spec_hi_m_idx = 0
        
        self.setImageROI()
        self.update_spectrum()
        self.update_image_roi()

        # image connections
        self.image_view.mousePressEvent = self.getPointSpectrum
        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.spec_roi_math.sigRegionChangeFinished.connect(self.spec_roi_calc)
        self.pb_apply_spec_calc.clicked.connect(self.spec_roi_calc)
        self.rb_math_roi.clicked.connect(self.update_spectrum)
        self.pb_add_roi_2.clicked.connect(self.math_img_roi_flag)
        self.image_roi_math.sigRegionChangeFinished.connect(self.image_roi_calc)
        self.pb_apply_img_calc.clicked.connect(self.image_roi_calc)

        [rbs.clicked.connect(self.setImageROI) for rbs in
         [self.rb_poly_roi, self.rb_elli_roi, self.rb_rect_roi,
          self.rb_line_roi, self.rb_circle_roi]]

    def select_elist(self):
        self.energyFileChooser()
        self.efileLoader()
        self.energyUnitCheck()
        self.view_stack()

    def efileLoader(self):

        if self.efilePath:

            if str(self.efilePath).endswith('log_tiff.txt'):
                self.energy = energy_from_logfile(logfile=str(self.efilePath))
                logger.info("Log file from pyxrf processing")

            else:
                self.energy = np.loadtxt(str(self.efilePath))
            self.change_color_on_load(self.pb_elist_xanes)
            logger.info('Energy file loaded')

        else:
            self.statusbar_main.showMessage("No Energy List Selected, Setting Arbitary Axis")
            self.energy = np.arange(self.im_stack.shape[0])
            logger.info('Arbitary Energy Axis')

        # assert len(self.energy) == self.dim1, "Number of Energy Points is not equal to stack length"

    def energyUnitCheck(self):

        if np.max(self.energy) < 100:
            self.cb_kev_flag.setChecked(True)
            self.energy *= 1000

        else:
            self.cb_kev_flag.setChecked(False)

    def select_ref_file(self):
        self.pb_xanes_fit.setEnabled(True)
        self.ref_names = []
        file_name = QFileDialog().getOpenFileName(self, "Open reference file", '', 'text file (*.txt *.nor)')
        if file_name[0]:
            if file_name[0].endswith('.nor'):
                self.refs, self.ref_names = create_df_from_nor_try2(athenafile=str(file_name[0]))
                self.change_color_on_load(self.pb_ref_xanes)

            elif file_name[0].endswith('.txt'):
                self.refs = pd.read_csv(str(file_name[0]), header=None, delim_whitespace=True)
                self.change_color_on_load(self.pb_ref_xanes)

        else:
            logger.error('No file selected')
            pass

        self.plt_xanes_refs()

    def plt_xanes_refs(self):

        try:
            self.ref_plot.close()

        except:
            pass

        self.ref_plot = plot(title="Reference Standards")
        self.ref_plot.setLabel("bottom", "Energy")
        self.ref_plot.setLabel("left", "Intensity")
        self.ref_plot.addLegend()

        for n in range(np.shape(self.refs)[1]):

            if not n == 0:
                self.ref_plot.plot(self.refs.values[:, 0], self.refs.values[:, n],
                                   pen=pg.mkPen(self.plt_colors[n - 1], width=self.plotWidth),
                                   name=self.ref_names[n])

    def getPointSpectrum(self, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.xpixel = int(self.image_view.view.mapSceneToView(event.pos()).x()) - 1
                zlim, ylim, xlim = self.displayedStack.shape

                if self.xpixel > xlim:
                    self.xpixel = xlim - 1

                self.ypixel = int(self.image_view.view.mapSceneToView(event.pos()).y()) - 1
                if self.ypixel > ylim:
                    self.ypixel = ylim - 1
                self.spectrum_view.addLegend()
                self.point_spectrum = self.displayedStack[:, self.ypixel, self.xpixel]
                self.spectrum_view.plot(self.xdata, self.point_spectrum, clear=True,
                                        pen=pg.mkPen(pg.mkColor(0, 0, 255, 255), width=self.plotWidth),
                                        symbol='o', symbolSize=6, symbolBrush='r',
                                        name=f'Point Spectrum; x= {self.xpixel}, y= {self.ypixel}')

                self.spectrum_view.addItem(self.spec_roi)

                self.statusbar_main.showMessage(f'{self.xpixel} and {self.ypixel}')

    def setImageROI(self):

        self.lineROI = pg.LineSegmentROI([[int(self.dim3 // 2), int(self.dim2 // 2)],
                                          [self.sz, self.sz]], pen='r')

        self.rectROI = pg.RectROI([int(self.dim3 // 2), int(self.dim2 // 2)],
                                  [self.sz, self.sz], pen='w', maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2))

        self.rectROI.addTranslateHandle([0, 0], [2, 2])
        self.rectROI.addRotateHandle([0, 1], [2, 2])

        self.ellipseROI = pg.EllipseROI([int(self.dim3 // 2), int(self.dim2 // 2)],
                                        [self.sz, self.sz], pen='w',
                                        maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2))

        self.circleROI = pg.CircleROI([int(self.dim3 // 2), int(self.dim2 // 2)],
                                      [self.sz, self.sz], pen='w',
                                      maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2))  # pos and size

        self.polyLineROI = pg.PolyLineROI([[0, 0], [0, self.sz], [self.sz, self.sz], [self.sz, 0]],
                                          pos=(int(self.dim3 // 2), int(self.dim2 // 2)),
                                          maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
                                          closed=True, removable=True)

        self.rois = {'rb_line_roi': self.lineROI, 'rb_rect_roi': self.rectROI, 'rb_circle_roi': self.circleROI,
                     'rb_elli_roi': self.ellipseROI, 'rb_poly_roi': self.polyLineROI}

        button_name = self.sender()

        if button_name.objectName() in self.rois.keys():
            self.roi_preference = button_name.objectName()

        else:
            self.roi_preference = 'rb_rect_roi'  # default

        try:
            self.image_view.removeItem(self.image_roi)

        except:
            pass

        # ROI settings for image, used polyline roi with non rectangular shape

        self.image_roi = self.rois[self.roi_preference]
        self.image_view.addItem(self.image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)

    def replot_image(self):
        self.update_stack()
        self.update_spectrum()
        self.update_image_roi()

    def update_spec_roi_values(self):
        self.stack_center = int(self.energy[len(self.energy) // 2])
        self.stack_width = int((self.energy.max() - self.energy.min()) * 0.05)
        self.spec_roi.setBounds([self.xdata[0], self.xdata[-1]])  # if want to set bounds for the spec roi
        self.spec_roi_math.setBounds([self.xdata[0], self.xdata[-1]])

    def update_spectrum(self):

        # set x-axis values; array taken from energy values, if clipped z box values will update the array
        self.xdata = self.energy[self.sb_zrange1.value():self.sb_zrange2.value()]

        # get the cropped stack from ROI region; pyqtgraph function is used
        self.roi_img_stk = self.image_roi.getArrayRegion(self.displayedStack, self.image_view.imageItem, axes=(1, 2))

        posx, posy = self.image_roi.pos()
        self.le_roi.setText(str(int(posx)) + ':' + str(int(posy)))

        # display the ROI features in the line edit boxes
        if self.roi_img_stk.ndim == 3:
            sizex, sizey = self.roi_img_stk.shape[1], self.roi_img_stk.shape[2]
            self.le_roi_size.setText(str(sizex) + ',' + str(sizey))
            self.mean_spectra = get_mean_spectra(self.roi_img_stk)

        elif self.roi_img_stk.ndim == 2:
            sizex, sizey = self.roi_img_stk.shape[0], self.roi_img_stk.shape[1]
            self.le_roi_size.setText(str(sizex) + ',' + str(sizey))
            self.mean_spectra = self.roi_img_stk.mean(-1)

        self.spectrum_view.addLegend()

        try:
            self.spectrum_view.plot(self.xdata, self.mean_spectra,
                                    pen=pg.mkPen(pg.mkColor(5, 255, 5, 255), width=self.plotWidth),
                                    clear=True, symbol='o', symbolSize=6, symbolBrush='r',
                                    name='ROI Spectrum')
        except:
            self.spectrum_view.plot(self.mean_spectra, clear=True, pen=pg.mkPen(pg.mkColor(5, 255, 5, 255),
                                                                                width=self.plotWidth),
                                    symbol='o', symbolSize=6, symbolBrush='r', name='ROI Spectrum')

        if self.energy[-1] > 1000:
            self.e_unit = 'eV'
        else:
            self.e_unit = 'keV'

        self.spectrum_view.setLabel('bottom', 'Energy', self.e_unit)
        self.spectrum_view.setLabel('left', 'Intensity', 'A.U.')
        self.spectrum_view.addItem(self.spec_roi)
        self.update_spec_roi_values()
        self.math_roi_flag()

    def update_image_roi(self):
        self.spec_lo, self.spec_hi = self.spec_roi.getRegion()
        self.spec_lo_idx = (np.abs(self.energy - self.spec_lo)).argmin()
        self.spec_hi_idx = (np.abs(self.energy - self.spec_hi)).argmin()
        self.le_spec_roi.setText(str(int(self.spec_lo)) + ':' + str(int(self.spec_hi)))
        self.le_spec_roi_size.setText(str(int(self.spec_hi - self.spec_lo)))
        self.update_spec_roi_values()
        self.stackIndexToNames()

        try:
            if int(self.spec_lo_idx) == int(self.spec_hi_idx):
                self.disp_img = self.displayedStack[int(self.spec_hi_idx), :, :]
                

            else:
                self.disp_img = self.displayedStack[int(self.spec_lo_idx):int(self.spec_hi_idx), :, :].sum(0)

            self.image_view.setImage(self.disp_img)
            self.statusbar_main.showMessage(f'Image Display is {self.corrImg1}')
        except:
            logger.warning("Indices are out of range; Image cannot be created")
            pass

    def set_spec_roi(self):
        self.spec_lo_, self.spec_hi_ = int(self.sb_roi_spec_s.value()), int(self.sb_roi_spec_e.value())
        self.spec_lo_idx_ = (np.abs(self.energy - self.spec_lo_)).argmin()
        self.spec_hi_idx_ = (np.abs(self.energy - self.spec_hi_)).argmin()
        self.spec_roi.setRegion((self.xdata[self.spec_lo_idx_], self.xdata[self.spec_hi_idx_]))
        self.update_image_roi()

    def math_roi_flag(self):
        if self.rb_math_roi.isChecked():
            self.rb_math_roi.setStyleSheet("color : green")
            self.spectrum_view.addItem(self.spec_roi_math)
        else:
            self.spectrum_view.removeItem(self.spec_roi_math)

    def spec_roi_calc(self):

        self.spec_lo_m, self.spec_hi_m = self.spec_roi_math.getRegion()
        self.spec_lo_m_idx = (np.abs(self.energy - self.spec_lo_m)).argmin()
        self.spec_hi_m_idx = (np.abs(self.energy - self.spec_hi_m)).argmin()

        if int(self.spec_lo_idx) == int(self.spec_hi_idx):
            self.img1 = self.displayedStack[int(self.spec_hi_idx), :, :]

        else:
            self.img1 = self.displayedStack[int(self.spec_lo_idx):int(self.spec_hi_idx), :, :].mean(0)

        if int(self.spec_lo_m_idx) == int(self.spec_hi_m_idx):
            self.img2 = self.displayedStack[int(self.spec_hi_m_idx), :, :]

        else:
            self.img2 = self.displayedStack[int(self.spec_lo_m_idx):int(self.spec_hi_m_idx), :, :].mean(0)

        if self.cb_roi_operation.currentText() == "Correlation Plot":
            self.correlation_plot()

        else:
            calc = {'Divide': np.divide, 'Subtract': np.subtract, 'Add': np.add}
            self.disp_img = remove_nan_inf(calc[self.cb_roi_operation.currentText()](self.img1, self.img2))
            self.image_view.setImage(self.disp_img)

    def math_img_roi_flag(self):

        button_name = self.sender().text()

        if button_name == 'Add ROI_2':
            self.image_view.addItem(self.image_roi_math)
            self.pb_add_roi_2.setText("Remove ROI_2")
            self.image_roi2_flag = 1
        elif button_name == 'Remove ROI_2':
            self.image_view.removeItem(self.image_roi_math)
            self.pb_add_roi_2.setText("Add ROI_2")
            self.image_roi2_flag = 0

        else:
            pass
            logger.error('Unknown signal for second ROI')

    def image_roi_calc(self):

        if self.image_roi2_flag == 1:
            self.calc = {'Divide': np.divide, 'Subtract': np.subtract, 'Add': np.add}
            self.update_spec_image_roi()
        else:
            logger.error("No ROI2 found")
            pass

    def update_spec_image_roi(self):

        self.math_roi_reg = self.image_roi_math.getArrayRegion(self.displayedStack,
                                                               self.image_view.imageItem, axes=(1, 2))
        if self.math_roi_reg.ndim == 3:

            self.math_roi_spectra = get_mean_spectra(self.math_roi_reg)

        elif self.roi_img_stk.ndim == 2:
            self.math_roi_spectra = self.math_roi_reg.mean(-1)

        if self.cb_img_roi_action.currentText() in self.calc.keys():

            calc_spec = self.calc[self.cb_img_roi_action.currentText()](self.mean_spectra,
                                                                        self.math_roi_spectra)
            self.spectrum_view.addLegend()
            self.spectrum_view.plot(self.xdata, calc_spec, clear=True, pen=pg.mkPen('m', width=2),
                                    name=self.cb_img_roi_action.currentText() + "ed")
            self.spectrum_view.plot(self.xdata, self.math_roi_spectra, pen=pg.mkPen('y', width=2),
                                    name="ROI2")
            self.spectrum_view.plot(self.xdata, self.mean_spectra, pen=pg.mkPen('g', width=2),
                                    name="ROI1")

        elif self.cb_img_roi_action.currentText() == 'Compare':
            self.spectrum_view.plot(self.xdata, self.math_roi_spectra, pen=pg.mkPen('y', width=2),
                                    clear=True, name="ROI2")
            self.spectrum_view.plot(self.xdata, self.mean_spectra, pen=pg.mkPen('g', width=2),
                                    name="ROI1")

        self.spectrum_view.addItem(self.spec_roi)

    def displayStackInfo(self):

        try:

            if isinstance(self.file_name, list):
                info = f'Folder; {os.path.dirname(self.file_name[0])} \n'
                for n, name in enumerate(self.file_name):
                    info += f'{n}: {os.path.basename(name)} \n'

                # info = f'Stack order; {[name for name in enumerate(self.file_name)]}'
            else:
                info = f'Stack; {self.file_name}'

            self.infoWindow = StackInfo(str(info))
            self.infoWindow.show()

        except AttributeError:
            self.statusbar_main.showMessage('Warning: No Image Data Loaded')

    def stackIndexToNames(self):
        #create list of tiff file names for virtutal stack for plot axes
        self.elemFileName = []

        if isinstance(self.file_name, list):
            for name in self.file_name:
                   self.elemFileName.append(os.path.basename(name).split('.')[0])

            logger.info(f" Virtual Stack - list of image names; {self.elemFileName}")
            
            #if the roi focus on one frame, Note that this slicing excludes the last index
            if int(self.spec_lo_idx) == int(self.spec_hi_idx):
                self.corrImg1 = str(self.elemFileName[int(self.spec_lo_idx)])
            else:
                self.corrImg1 = self.elemFileName[int(self.spec_lo_idx):int(self.spec_hi_idx)]
                if len(self.corrImg1)>1:
                   self.corrImg1 = f"Sum of {self.corrImg1} " 

            if int(self.spec_lo_m_idx) == int(self.spec_hi_m_idx):
                self.corrImg2 = str(self.elemFileName[int(self.spec_lo_m_idx)])

            else:
                self.corrImg2 = self.elemFileName[int(self.spec_lo_m_idx):int(self.spec_hi_m_idx)]

                if len(self.corrImg2)>1:
                    self.corrImg2 = f"Sum of {self.corrImg2}"

            logger.info(f'Correlation stack {int(self.spec_lo_idx)}:{int(self.spec_hi_idx)} with {int(self.spec_lo_m_idx)}:{int(self.spec_hi_m_idx)}')

            logger.info(f" Virtual Stack; corrlation plot of {self.corrImg1} vs {self.corrImg2}")
        else:
            self.corrImg1 =f" Sum of {os.path.basename(self.file_name).split('.')[0]}_{int(self.spec_lo_idx)} to {int(self.spec_hi_idx)}"
            self.corrImg2 = f" Sum of {os.path.basename(self.file_name).split('.')[0]}_{int(self.spec_lo_m_idx)} to {int(self.spec_hi_m_idx)}"
            #logger.info(f" corrlation plot of {self.corrImg1} vs {self.corrImg2}")

    def correlation_plot(self):
        self.stackIndexToNames()

        self.statusbar_main.showMessage(f'Correlation of {self.corrImg1} with {self.corrImg2}')
        
        self.scatter_window = ScatterPlot(self.img1, self.img2, (str(self.corrImg1),str(self.corrImg2)))

        ph = self.geometry().height()
        pw = self.geometry().width()
        px = self.geometry().x()
        py = self.geometry().y()
        dw = self.scatter_window.width()
        dh = self.scatter_window.height()
        # self.scatter_window.setGeometry(px+0.65*pw, py + ph - 2*dh-5, dw, dh)
        self.scatter_window.show()

    def getROIMask(self):
        self.roi_mask = self.image_roi.getArrayRegion(self.displayedStack, self.image_view.imageItem,
                                                      axes=(1, 2))
        self.newWindow = singleStackViewer(self.roi_mask)
        self.newWindow.show()

    def save_stack(self, saveSum=False):

        # self.update_stack()
        file_name = QFileDialog().getSaveFileName(self, "Save image data", 'image_data.tiff', 'image file(*tiff *tif )')
        if file_name[0]:
            if not saveSum:
                tf.imsave(str(file_name[0]), self.displayedStack)
                logger.info(f'Updated Image Saved: {str(file_name[0])}')
                self.statusbar_main.showMessage(f'Updated Image Saved: {str(file_name[0])}')
            if saveSum:
                tf.imsave(str(file_name[0]), np.sum(self.displayedStack, axis=0))

        else:
            self.statusbar_main.showMessage('Saving cancelled')
            pass

    def save_disp_img(self):
        file_name = QFileDialog().getSaveFileName(self, "Save image data", 'image.tiff', 'image file(*tiff *tif )')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '.tiff', self.disp_img.T)
            self.statusbar_main.showMessage(f'Image Saved to {str(file_name[0])}')
            logger.info(f'Updated Image Saved: {str(file_name[0])}')

        else:
            logger.error('No file to save')
            self.statusbar_main.showMessage('Saving cancelled')
            pass

    def getLivePlotData(self):
        try:

            data = np.squeeze([c.getData() for c in self.spectrum_view.plotItem.curves])
            # print(np.shape(data))
            if data.ndim == 2:
                self.mu_ = data[1]
                self.e_ = data[0]
            elif data.ndim == 3:
                e_mu = data[0, :, :]
                self.mu_ = e_mu[1]
                self.e_ = e_mu[0]

            else:
                logger.error(f" Data shape of {data.ndim} is not supported")
                pass
        except AttributeError:
            logger.error("No data loaded")
            pass

    def addSpectrumToCollector(self):
        self.getLivePlotData()
        self.spectrum_view_collect.plot(self.e_, self.mu_, name='ROI Spectrum')
        self.spectrum_view_collect.setLabel('bottom', 'Energy', self.e_unit)
        self.spectrum_view_collect.setLabel('left', 'Intensity', 'A.U.')

    def findEo(self):
        try:
            self.getLivePlotData()
            e0_init = self.e_[np.argmax(np.gradient(self.mu_))]
            self.dsb_norm_Eo.setValue(e0_init)

        except AttributeError:
            logger.error("No data loaded")
            pass

    def initNormVals(self):
        self.getLivePlotData()
        e0_init = self.e_[np.argmax(np.gradient(self.mu_))]
        pre1, pre2, post1, post2 = xanesNormalization(self.e_, self.mu_, e0=e0_init, step=None,
                                                      nnorm=1, nvict=0, guess=True)
        self.dsb_norm_pre1.setValue(pre1)
        self.dsb_norm_pre2.setValue(pre2)
        self.dsb_norm_post1.setValue(post1)
        self.dsb_norm_post2.setValue(post2)
        self.dsb_norm_Eo.setValue(e0_init)

    def getNormParams(self):
        self.getLivePlotData()
        eo_ = self.dsb_norm_Eo.value()
        pre1_, pre2_ = self.dsb_norm_pre1.value(), self.dsb_norm_pre2.value()
        norm1_, norm2_ = self.dsb_norm_post1.value(), self.dsb_norm_post2.value()
        norm_order = self.sb_norm_order.value()

        return eo_, pre1_, pre2_, norm1_, norm2_, norm_order

    def exportNormParams(self):
        self.xanesNormParam = {}
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()
        self.xanesNormParam['E0'] = eo_
        self.xanesNormParam['pre1'] = pre1_
        self.xanesNormParam['pre2'] = pre2_
        self.xanesNormParam['post1'] = norm1_
        self.xanesNormParam['post2'] = norm2_
        self.xanesNormParam['norm_order'] = norm_order

        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save XANES Norm Params", 'xanes_norm_params.csv',
                                                            'csv file(*csv)')

        if file_name[0]:

            pd.DataFrame(self.xanesNormParam, index=[0]).to_csv(file_name[0])

        else:
            pass

    def importNormParams(self):

        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open a XANES Norm File", '',
                                                            'csv file(*csv);;all_files (*)')

        if file_name[0]:
            xanesNormParam = pd.read_csv(file_name[0])
            self.dsb_norm_Eo.setValue(xanesNormParam["E0"])
            self.dsb_norm_pre1.setValue(xanesNormParam["pre1"])
            self.dsb_norm_pre2.setValue(xanesNormParam["pre2"])
            self.dsb_norm_post1.setValue(xanesNormParam["post1"])
            self.dsb_norm_post2.setValue(xanesNormParam["post2"])
            self.sb_norm_order.setValue(xanesNormParam["norm_order"])

    def nomalizeLiveSpec(self):
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()
        self.spectrum_view.clear()

        pre_line, post_line, normXANES = xanesNormalization(self.e_, self.mu_, e0=eo_, step=None,
                                                            nnorm=norm_order, nvict=0, pre1=pre1_, pre2=pre2_,
                                                            norm1=norm1_, norm2=norm2_)

        names = np.array(('Spectrum', 'Pre', 'Post'))
        data_array = np.array((self.mu_, pre_line, post_line))
        colors = np.array(('c', 'r', 'm'))

        for data, clr, name in zip(data_array, colors, names):
            self.spectrum_view.plot(self.e_, data, pen=pg.mkPen(clr, width=self.plotWidth), name=name)

        self.spectrum_view_norm.plot(self.e_, normXANES, clear=True,
                                     pen=pg.mkPen(self.plt_colors[-1], width=self.plotWidth))
        self.spectrum_view_norm.setLabel('bottom', 'Energy', self.e_unit)
        self.spectrum_view_norm.setLabel('left', 'Norm. Intensity', 'A.U.')

    def normalizeStack(self):
        self.getLivePlotData()
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()

        self.im_stack = self.displayedStack = xanesNormStack(self.e_, self.displayedStack, e0=eo_, step=None,
                                            nnorm=norm_order, nvict=0, pre1=pre1_, pre2=pre2_,
                                            norm1=norm1_, norm2=norm2_)
        #self.im_stack = self.displayedStack

    def transposeStack(self):
        self.im_stack = self.displayedStack = np.transpose(self.displayedStack, (2,1,0))

    def swapStackXY(self):
        self.im_stack = self.displayedStack = np.transpose(self.displayedStack, (0,2,1))

    def removeROIBGStack(self):
        self.displayedStack = subtractBackground(self.displayedStack, self.mean_spectra)

    def resetCollectorSpec(self):
        pass

    def saveCollectorPlot(self):
        exporter = pg.exporters.CSVExporter(self.spectrum_view_collect.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectra", '', 'spectra (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]) + '.csv')
        else:
            self.statusbar_main.showMessage('Saving cancelled')
            pass

    def save_disp_spec(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectrum", '', 'spectra (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]) + '.csv')
        else:
            self.statusbar_main.showMessage('Saving cancelled')
            pass

    def saveEnergyList(self):
        file_name = QFileDialog().getSaveFileName(self, "save energy list", 'energy_list.txt', 'text file (*txt)')
        if file_name[0]:
            np.savetxt(file_name[0], self.xdata, fmt='%.4f')
        else:
            pass

    def pca_scree_(self):
        logger.info('Process started..')
        self.update_stack()
        pca_scree(self.displayedStack)
        logger.info('Process complete')

    def calc_comp_(self):

        logger.info('Process started..')

        # self.update_stack()
        n_components = self.sb_ncomp.value()
        method_ = self.cb_comp_method.currentText()

        ims, comp_spec, decon_spec, decomp_map = decompose_stack(self.displayedStack,
                                                                 decompose_method=method_, n_components_=n_components)

        self._new_window3 = ComponentViewer(ims, self.xdata, comp_spec, decon_spec, decomp_map)
        self._new_window3.show()

        logger.info('Process complete')

    def kmeans_elbow(self):
        logger.info('Process started..')
        # self.update_stack()

        with pg.BusyCursor():
            try:
                kmeans_variance(self.displayedStack)
                logger.info('Process complete')
            except OverflowError:
                pass
                logger.error('Overflow Error, values are too long')

    def kmeans_elbow_Thread(self):
        # Pass the function to execute
        worker = Worker(self.kmeans_elbow)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # Execute
        self.threadpool.start(worker)

    def clustering_(self):

        logger.info('Process started..')
        # self.update_stack()
        method_ = self.cb_clust_method.currentText()

        decon_images, X_cluster, decon_spectra = cluster_stack(self.displayedStack, method=method_,
                                                               n_clusters_=self.sb_ncluster.value(),
                                                               decomposed=False,
                                                               decompose_method=self.cb_comp_method.currentText(),
                                                               decompose_comp=self.sb_ncomp.value())

        self._new_window4 = ClusterViewer(decon_images, self.xdata, X_cluster, decon_spectra)
        self._new_window4.show()

        logger.info('Process complete')

    def change_color_on_load(self, button_name):
        button_name.setStyleSheet("background-color : rgb(0,150,0);"
                                  "color: rgb(255,255,255)")

    def energyFileChooser(self):
        file_name = QFileDialog().getOpenFileName(self, "Open energy list", '', 'text file (*.txt)')
        self.efilePath = file_name[0]

    def fast_xanes_fitting(self):

        self._new_window5 = XANESViewer(self.displayedStack, self.xdata, self.refs, self.ref_names)
        self._new_window5.show()

    # Thread Signals

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            QApplication.closeAllWindows()
        else:
            event.ignore()

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    '''
    start = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        self.signals.start.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class singleStackViewer(QtWidgets.QMainWindow):
    def __init__(self, img_stack, gradient='viridis'):
        super(singleStackViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_path, 'uis/singleStackView.ui'), self)

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
        self.hs_img_stack.setMaximum(self.dim1 - 1)
        self.hs_img_stack.setValue(np.round(self.dim1 / 2))
        self.displayStack()

        # connections
        self.hs_img_stack.valueChanged.connect(self.displayStack)
        self.actionSave.triggered.connect(self.saveImageStackAsTIFF)

    def displayStack(self):
        im_index = self.hs_img_stack.value()
        if self.img_stack.ndim == 2:
            self.image_view.setImage(self.img_stack)
        else:
            self.image_view.setImage(self.img_stack[im_index])
        self.label_img_count.setText(f'{im_index + 1}/{self.dim1}')

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

    def __init__(self, comp_stack, energy, comp_spectra, decon_spectra, decomp_map):
        super(ComponentViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_path, 'uis/ComponentView.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())

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
        self.pb_openScatterPlot.clicked.connect(self.openScatterPlot)
        self.pb_showMultiColor.clicked.connect(self.generateMultiColorView)

    def update_image(self):
        im_index = self.hs_comp_number.value()
        self.spectrum_view.setLabel('bottom', 'Energy')
        self.spectrum_view.setLabel('left', 'Intensity', 'A.U.')
        self.spectrum_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        self.component_view.setLabel('bottom', 'Energy')
        self.component_view.setLabel('left', 'Weight', 'A.U.')
        self.component_view.plot(self.energy, self.comp_spectra[:, im_index], clear=True)
        self.label_comp_number.setText(f'{im_index + 1}/{self.dim1}')
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])

    def openScatterPlot(self):
        self.scatter_window = ComponentScatterPlot(self.comp_stack, self.comp_spectra)

        ph = self.geometry().height()
        pw = self.geometry().width()
        px = self.geometry().x()
        py = self.geometry().y()
        dw = self.scatter_window.width()
        dh = self.scatter_window.height()
        # self.scatter_window.setGeometry(px+0.65*pw, py + ph - 2*dh-5, dw, dh)
        self.scatter_window.show()

    def show_all_spec(self):
        self.spectrum_view.clear()
        self.plt_colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w'] * 10
        offsets = np.arange(0, 2, 0.2)
        self.spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.spectrum_view.plot(self.energy,
                                    (self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                                    pen=self.plt_colors[ii], name="component" + str(ii + 1))

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '_components.tiff', np.float32(self.comp_stack.transpose(0, 2, 1)),
                      imagej=True)
            tf.imsave(str(file_name[0]) + '_component_masks.tiff', np.float32(self.decomp_map.T), imagej=True)
            np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)
            np.savetxt(str(file_name[0]) + '_component_spec.txt', self.comp_spectra)
        else:
            pass

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.comp_stack.transpose(0, 1, 2))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f'Image {n + 1}'] = {'ImageName': f'Image {n + 1}',
                                                 'ImageDir': '.',
                                                 'Image': image,
                                                 'Color': colorName,
                                                 'CmapLimits': (low, high),
                                                 'Opacity': 1.0
                                                 }
        self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)
        self.muli_color_window.show()

    # add energy column

class ClusterViewer(QtWidgets.QMainWindow):

    def __init__(self, decon_images, energy, X_cluster, decon_spectra):
        super(ClusterViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_path, 'uis/ClusterView.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())

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
        self.pb_show_all_spec.clicked.connect(self.showAllSpec)
        self.pb_showMultiColor.clicked.connect(self.generateMultiColorView)

    def update_display(self):
        im_index = self.hsb_cluster_number.value()
        self.component_view.setLabel('bottom', 'Energy')
        self.component_view.setLabel('left', 'Intensity', 'A.U.')
        self.component_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.decon_images[im_index])
        self.label_comp_number.setText(f'{im_index + 1}/{self.dim1}')

    def save_clust_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", '', 'data(*tiff *tif *txt *png )')
        if file_name[0]:

            tf.imsave(str(file_name[0]) + '_cluster.tiff', np.float32(self.decon_images.transpose(0, 2, 1)),
                      imagej=True)
            tf.imsave(str(file_name[0]) + '_cluster_map.tiff', np.float32(self.X_cluster.T), imagej=True)
            np.savetxt(str(file_name[0]) + '_deconv_spec.txt', self.decon_spectra)

        else:
            logger.error("Saving Cancelled")
            self.statusbar.showMessage("Saving Cancelled")
            pass

    def showAllSpec(self):
        self.component_view.clear()
        self.plt_colors = ['g', 'b', 'r', 'c', 'm', 'y', 'w'] * 10
        offsets = np.arange(0, 2, 0.2)
        self.component_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.component_view.plot(self.energy,
                                    (self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                                    pen=self.plt_colors[ii], name="cluster" + str(ii + 1))

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.decon_images.transpose(0, 1, 2))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f'Image {n + 1}'] = {'ImageName': f'Image {n + 1}',
                                                 'ImageDir': '.',
                                                 'Image': image,
                                                 'Color': colorName,
                                                 'CmapLimits': (low, high),
                                                 'Opacity': 1.0
                                                 }
        self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)
        self.muli_color_window.show()

class XANESViewer(QtWidgets.QMainWindow):

    def __init__(self, im_stack=None, e_list=None, refs=None, ref_names=None):
        super(XANESViewer, self).__init__()

        uic.loadUi(os.path.join(ui_path, 'uis/XANESViewer.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())

        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs
        self.ref_names = ref_names
        self.selected = self.ref_names
        self.fitResultDict = {}
        self.fit_method = self.cb_xanes_fit_model.currentText()

        self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack, self.e_list,
                                                                      self.refs, method=self.fit_method)

        (self.dim1, self.dim2, self.dim3) = self.im_stack.shape
        self.cn = int(self.dim2 // 2)
        self.sz = np.max([int(self.dim2 * 0.15), int(self.dim3 * 0.15)])
        self.image_roi = pg.RectROI([int(self.dim3 // 2), int(self.dim2 // 2)],
                                    [self.sz, self.sz], pen='w', maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2))

        self.image_roi.addTranslateHandle([0, 0], [2, 2])
        self.image_roi.addRotateHandle([0, 1], [2, 2])

        #self.image_roi = pg.PolyLineROI([[0, 0], [0, self.sz], [self.sz, self.sz], [self.sz, 0]],
                                        #pos=(int(self.dim2 // 2), int(self.dim3 // 2)),
                                        #maxBounds=QtCore.QRect(0, 0, self.dim3, self.dim2), closed=True)
        #self.image_roi.addTranslateHandle([self.sz // 2, self.sz // 2], [2, 2])

        self.stack_center = int(self.dim1 // 2)
        self.stack_width = int(self.dim1 * 0.05)
        # self.image_view.setCurrentIndex(self.stack_center)

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
        self.hsb_xanes_stk.valueChanged.connect(self.display_image_data)
        self.hsb_chem_map.valueChanged.connect(self.display_image_data)
        self.pb_showMultiColor.clicked.connect(self.generateMultiColorView)

        #menu
        self.actionSave_Chem_Map.triggered.connect(self.save_chem_map)
        self.actionSave_R_factor_Image.triggered.connect(self.save_rfactor_img)
        self.actionSave_Live_Fit_Data.triggered.connect(self.pg_export_spec_fit)
        self.actionExport_Fit_Stats.triggered.connect(self.exportFitResults)
        self.actionExport_Ref_Plot.triggered.connect(self.pg_export_references)

    def scrollBar_setup(self):
        self.hsb_xanes_stk.setValue(self.stack_center)
        self.hsb_xanes_stk.setMaximum(self.dim1 - 1)
        self.hsb_chem_map.setValue(0)
        self.hsb_chem_map.setMaximum(self.decon_ims.shape[-1] - 1)

    def display_image_data(self):

        self.image_view.setImage(self.im_stack[self.hsb_xanes_stk.value()])
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('viridis')

        self.image_view_maps.setImage(self.decon_ims.transpose(2, 0, 1)[self.hsb_chem_map.value()])
        self.image_view_maps.setPredefinedGradient('bipolar')
        self.image_view_maps.ui.menuBtn.hide()
        self.image_view_maps.ui.roiBtn.hide()

    def display_references(self):

        self.inter_ref = interploate_E(self.refs, self.xdata)
        self.plt_colors = ['c', 'm', 'y', 'w'] * 10
        self.spectrum_view_refs.addLegend()
        for ii in range(self.inter_ref.shape[0]):
            if len(self.selected) != 0:
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=pg.mkPen(self.plt_colors[ii], width=2),
                                             name=self.selected[1:][ii])
            else:
                self.spectrum_view_refs.plot(self.xdata, self.inter_ref[ii], pen=pg.mkPen(self.plt_colors[ii], width=2),
                                             name="ref" + str(ii + 1))

    def choose_refs(self):
        'Interactively exclude some standards from the reference file'
        self.ref_edit_window = RefChooser(self.ref_names, self.im_stack, self.e_list,
                                          self.refs, self.sb_e_shift.value(),
                                          self.cb_xanes_fit_model.currentText())
        self.ref_edit_window.show()
        # self.rf_plot = pg.plot(title="RFactor Tracker")

        # connections
        self.ref_edit_window.choosenRefsSignal.connect(self.update_refs)
        self.ref_edit_window.fitResultsSignal.connect(self.plotFitResults)

    def update_refs(self, list_):
        self.selected = list_  # list_ is the signal from ref chooser
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
        self.spectrum_view.setLabel('bottom', 'Energy')
        self.spectrum_view.setLabel('left', 'Intensity', 'A.U.')
        self.spectrum_view.plot(self.xdata1, self.ydata1, pen=pen, name="Data", clear=True)
        self.spectrum_view.plot(self.xdata1, self.fit_, name="Fit", pen=pen2)

        for n, (coff, ref, plt_clr) in enumerate(zip(coeffs, self.inter_ref, self.plt_colors)):

            if len(self.selected) != 0:

                self.spectrum_view.plot(self.xdata1, np.dot(coff, ref), name=self.selected[1:][n], pen=plt_clr)
            else:
                self.spectrum_view.plot(self.xdata1, np.dot(coff, ref), name="ref" + str(n + 1), pen=plt_clr)
        # set the rfactor value to the line edit slot
        self.results = f"Coefficients: {coeffs} \n" \
                       f"R-Factor: {stats['R_Factor']}, R-Square: {stats['R_Square']},\n " \
                       f"Chi-Square: {stats['Chi_Square']}, " \
                       f"Reduced Chi-Square: {stats['Reduced Chi_Square']}"

        self.fit_results.setText(self.results)

    def re_fit_xanes(self):
        if len(self.selected) != 0:
            self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack,
                                                                          self.e_list + self.sb_e_shift.value(),
                                                                          self.refs[self.selected],
                                                                          method=self.cb_xanes_fit_model.currentText())
        else:
            # if non athena file with no header is loaded no ref file cannot be edited
            self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack,
                                                                          self.e_list + self.sb_e_shift.value(),
                                                                          self.refs,
                                                                          method=self.cb_xanes_fit_model.currentText())

        # rfactor is a list of all spectra so take the mean
        self.rfactor_mean = np.mean(self.rfactor)
        self.image_view_maps.setImage(self.decon_ims.transpose(2, 0, 1))
        self.scrollBar_setup()

    def plotFitResults(self, decon_ims, rfactor_mean, coeff_array):
        # upadte the chem maps and scrollbar params
        self.image_view_maps.setImage(decon_ims.transpose(2, 0, 1))
        # self.hsb_chem_map.setValue(0)
        # self.hsb_chem_map.setMaximum(decon_ims.shape[-1]-1)

        # set the rfactor value to the line edit slot
        self.le_r_sq.setText(f'{rfactor_mean :.4f}')

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.decon_ims.transpose((2,0,1)))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f'Image {n + 1}'] = {'ImageName': f'Image {n + 1}',
                                                 'ImageDir': '.',
                                                 'Image': image,
                                                 'Color': colorName,
                                                 'CmapLimits': (low, high),
                                                 'Opacity': 1.0
                                                 }
        self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)
        self.muli_color_window.show()

    def save_chem_map(self):
        file_name = QFileDialog().getSaveFileName(self, "save image", 'chemical_map.tiff', 'image data (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]) , np.float32(self.decon_ims.transpose(2,0,1)), imagej=True)
        else:
            logger.error('No file to save')
            pass

    def save_rfactor_img(self):
        file_name = QFileDialog().getSaveFileName(self, "save image", 'r-factor_map.tiff', 'image data (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]), np.float32(self.rfactor), imagej=True)
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
            exporter.export(str(file_name[0]) + '.csv')
        else:
            pass

    def pg_export_references(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view_refs.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save references", 'xanes_references.csv', 'column data (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]))
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

    def __init__(self, ref_names, im_stack, e_list, refs, e_shift, fit_model):
        super(RefChooser, self).__init__()
        uic.loadUi(os.path.join(ui_path, 'uis/RefChooser.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())
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
        self.stat_view.setLabel('left', 'Reduced Chi^2')

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

        # connections
        self.pb_apply.clicked.connect(self.clickedWhichAre)
        self.pb_combo.clicked.connect(self.tryAllCombo)
        self.actionExport_Results_csv.triggered.connect(self.exportFitResults)
        self.selectionLine.sigPositionChanged.connect(self.updateFitWithLine)
        self.tableWidget.itemSelectionChanged.connect(self.updateWithTableSelection)
        # self.stat_view.scene().sigMouseClicked.connect(self.moveSelectionLine)
        self.stat_view.mouseDoubleClickEvent = self.moveSelectionLine
        self.sb_max_combo.valueChanged.connect(self.displayCombinations)
        # self.pb_sort_with_r.clicked.connect(lambda: self.tableWidget.sortItems(3, QtCore.Qt.AscendingOrder))
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

    def generateRefList(self, ref_list, maxCombo, minCombo=1):
        
        """
        Creates a list of reference combinations for xanes fitting
        
        Paramaters;
        
        ref_list (list): list of ref names from the header
        maxCombo (int): maximum number of ref lists in combination
        minCombo (int): min number of ref lists in combination
        
        returns;

        1. int: length of total number of combinations
        2. list: all the combinations

        """
        
        if not maxCombo>len(ref_list):

            iter_list = []
            while minCombo < maxCombo + 1:
                iter_list += list(combinations(ref_list, minCombo))
                minCombo += 1
            return len(iter_list), iter_list

        else: raise ValueError(" Maximum numbinations cannot be larger than number of list items")

    def displayCombinations(self):
        niter, self.iter_list = self.generateRefList(self.ref_names[1:], self.sb_max_combo.value())
        self.label_nComb.setText(str(niter) + " Combinations")

    @QtCore.pyqtSlot()
    def tryAllCombo(self):
        #empty list to to keep track and plot of reduced chi2 of all the fits 
        self.rfactor_list = []
        
        #create dataframe for the table
        self.df = pd.DataFrame(columns=['Fit Number', 'References', 'Coefficients',
                                        'R-Factor', 'R^2', 'chi^2', 'red-chi^2', 'Score'])
        
        #df columns is the header for the table widget
        self.tableWidget.setHorizontalHeaderLabels(self.df.columns)
        # self.iter_list = list(combinations(self.ref_names[1:],self.sb_max_combo.value()))
        
        niter, self.iter_list = self.generateRefList(self.ref_names[1:], self.sb_max_combo.value())
        tot_combo = len(self.iter_list)
        for n, refs in enumerate(self.iter_list):
            self.statusbar.showMessage(f"{n + 1}/{tot_combo}")
            selectedRefs = (list((str(self.ref_names[0]),) + refs))
            self.fit_combo_progress.setValue((n + 1) * 100 / tot_combo)
            self.stat, self.coeffs_arr = xanes_fitting_Binned(self.im_stack, self.e_list + self.e_shift,
                                                            self.refs[selectedRefs], method=self.fit_model)

            self.rfactor_list.append(self.stat['Reduced Chi_Square'])
            self.stat_view.plot(x=np.arange(n + 1), y=self.rfactor_list, clear=True, title='Reduced Chi^2',
                                pen=pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')
            
            #arbitary number to rank the best fit
            fit_score = (self.stat['R_Square']+np.sum(self.coeffs_arr))/(self.stat['R_Factor']+
                                                                            self.stat['Reduced Chi_Square'])

            resultsDict = {'Fit Number': n, 'References': str(selectedRefs[1:]),
                           'Coefficients': str(np.around(self.coeffs_arr, 4)),
                           'Sum of Coefficients' :str(np.around(np.sum(self.coeffs_arr),4)),
                           'R-Factor': self.stat['R_Factor'], 'R^2': self.stat['R_Square'],
                           'chi^2': self.stat['Chi_Square'], 'red-chi^2': self.stat['Reduced Chi_Square'],
                           'Score': np.around(fit_score,4)}

            self.df = pd.concat([self.df, pd.DataFrame([resultsDict])], ignore_index=True)

            self.dataFrametoQTable(self.df)
            QtTest.QTest.qWait(0.1)  # hepls with real time plotting

        self.stat_view.addItem(self.selectionLine)

    def dataFrametoQTable(self, df_: pd.DataFrame):
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

    def selectTableAndCheckBox(self, x):
        nSelection = int(round(x))
        self.tableWidget.selectRow(nSelection)
        fit_num = int(self.tableWidget.item(nSelection, 0).text())
        refs_selected = self.iter_list[fit_num]

        # reset all the checkboxes to uncheck state, except the energy
        for checkstate in self.findChildren(QtWidgets.QCheckBox):
            if checkstate.isEnabled():
                checkstate.setChecked(False)

        for cb_names in refs_selected:
            checkbox = self.findChild(QtWidgets.QCheckBox, name=cb_names)
            checkbox.setChecked(True)

    def updateFitWithLine(self):
        pos_x, pos_y = self.selectionLine.pos()
        x = self.df.index[self.df[str('Fit Number')] == np.round(pos_x)][0]
        self.selectTableAndCheckBox(x)

    def updateWithTableSelection(self):
        x = self.tableWidget.currentRow()
        self.selectTableAndCheckBox(x)

    def moveSelectionLine(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            Pos = self.stat_view.plotItem.vb.mapSceneToView(event.pos())
            self.selectionLine.setPos(Pos.x())

    def sortTable(self):
        sorter_dict = {'R-Factor': 'R-Factor', 'R-Square': 'R^2',
                       'Chi-Square': 'chi^2', 'Reduced Chi-Square': 'red-chi^2',
                       'Fit Number': 'Fit Number'}
        sorter = sorter_dict[self.cb_sorter.currentText()]
        self.df = self.df.sort_values(sorter, ignore_index=True)
        self.dataFrametoQTable(self.df)

    def enableApply(self):

        """  """
        self.populateChecked()
        if len(self.onlyCheckedBoxes) > 1:
            self.pb_apply.setEnabled(True)
        else:
            self.pb_apply.setEnabled(False)

class ScatterPlot(QtWidgets.QMainWindow):

    def __init__(self, img1, img2, nameTuple):
        super(ScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_path, 'uis/ScatterView.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())
        self.clearPgPlot()
        self.w1 = self.scatterViewer.addPlot()
        self.img1 = img1
        self.img2 = img2
        self.nameTuple = nameTuple
        x, y = np.shape(self.img1)
        self.s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,0, 255))
        print(self.s1)
        #create three polyline ROIs for masking
        self.Xsize = self.img1.max() / 10
        self.Ysize = self.img2.max() / 10
        self.pos = self.img1.flatten().mean()
        self.scatter_mask = pg.PolyLineROI([[0, 0], [0, self.Ysize], [self.Xsize, self.Ysize], [self.Xsize, 0]],
                                           pos=None, pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('w', width=2), 
                                           closed=True, removable=True)
        self.scatter_mask2 = pg.PolyLineROI([[self.Xsize*1.2, 0], [self.Xsize*1.2, self.Ysize], [self.Xsize*2, self.Ysize], [self.Xsize*2, 0]],
                                           pos=None, pen=pg.mkPen('g', width=2), hoverPen=pg.mkPen('w', width=2), 
                                           closed=True, removable=True)
        self.scatter_mask3 = pg.PolyLineROI([[self.Xsize*2.5, 0], [self.Xsize*2.5, self.Ysize], [self.Xsize*4, self.Ysize], [self.Xsize*4, 0]],
                                           pos=None, pen=pg.mkPen('c', width=2),hoverPen=pg.mkPen('w', width=2), 
                                           closed=True, removable=True)

        self.fitScatter = self.fitScatter2 = self.fitScatter3 = None
                                           
        self.rois = {
                    'ROI 1':(self.scatter_mask,self.rb_roi1.isChecked(),self.fitScatter),
                    'ROI 2':(self.scatter_mask2,self.rb_roi2.isChecked(),self.fitScatter2),
                    'ROI 3':(self.scatter_mask3,self.rb_roi3.isChecked(),self.fitScatter3)
                    }

        self.windowNames = {
                    'ROI 1':self.fitScatter,
                    'ROI 2':self.fitScatter2,
                    'ROI 3':self.fitScatter3
                    }

        self.s1.setData(self.img1.flatten(), self.img2.flatten())
        self.w1.setLabel('bottom', self.nameTuple[0], 'counts')
        self.label_img1.setText(self.nameTuple[0])
        self.w1.setLabel('left', self.nameTuple[1], 'counts')
        self.label_img2.setText(self.nameTuple[1])
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
        #self.pb_define_mask.clicked.connect(lambda:self.createMask(self.scatter_mask))
        self.pb_define_mask.clicked.connect(self.addMultipleROIs)
        #self.pb_apply_mask.clicked.connect(lambda:self.getMaskRegion(self.scatter_mask))
        self.pb_apply_mask.clicked.connect(self.applyMultipleROIs)
        self.pb_clear_mask.clicked.connect(self.clearMultipleROIs)
        self.pb_compositeScatter.clicked.connect(self.createCompositeScatter)
        [rbs.clicked.connect(self.updateROIDict) for rbs in
         [self.rb_roi1, self.rb_roi2, self.rb_roi3]]

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.w1)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save correlation", '', 'spectrum and fit (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]) + '.csv')
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            pass

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, "save images", '', 'spectrum and fit (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '.tiff', np.dstack([self.img1, self.img2]).T)
            self.statusbar.showMessage(f"Images saved to {str(file_name[0])}")
        else:
            pass

    def createMask(self, ROIName):
        
        try: self.w1.removeItem(ROIName)
        except: pass
        self.w1.addItem(ROIName)

    def clearMask(self, ROIName):
        self.w1.removeItem(ROIName)

    def clearPgPlot(self):
        try:
            self.masked_img.close()
        except:
            pass

    def getMaskRegion(self, ROIName, generateSeperateWindows = True):

        """ filter scatterplot points using polylineROI region """

        # Ref : https://stackoverflow.com/questions/57719303/how-to-map-mouse-position-on-a-scatterplot

        #get the roi region:QPaintPathObject
        roiShape = self.rois[ROIName][0].mapToItem(self.s1, self.rois[ROIName][0].shape())
        
        #get data in the scatter plot
        scatterData = np.array(self.s1.getData())
        
        #generate a binary mask for points inside or outside the roishape
        selected = [roiShape.contains(QtCore.QPointF(pt[0],pt[1])) for pt in scatterData.T]

        # reshape the mask to image dimensions
        img_selected = np.reshape(selected, (self.img1.shape))
        
        #get masked image1
        self.maskedImage = img_selected * self.img1

        #get rid of the (0,0) values in the masked array
        self.xData, self.yData = np.compress(selected,scatterData[0]), np.compress(selected,scatterData[1])
        
        #linear regeression of the filtered X,Y data
        result = linregress(self.xData, self.yData)

        #Pearson's correlation of the filtered X,Y data
        pr, pp = stats.pearsonr(self.xData, self.yData)

        #apply the solved equation to xData to generate the fit line
        self.yyData = result.intercept + result.slope*self.xData

        #Prepare strings for fit results and stats
        self.fitLineEqn = f' y =  x*{result.slope :.3e} + {result.intercept :.3e}, R^2 = {result.rvalue**2 :.3f}, r = {pr :.3f}\n'
        FitStats1 = f' Slope Error = {result.stderr :.3e}, Intercept Error = {result.intercept_stderr :.3e}\n'
        FitStats2 = f' Pearson’s correlation coefficient = {pr :.3f}'
        refs = '\n\n ***References****\n\n scipy.stats.linregress, scipy.stats.pearsonr '
        fitStats = f'\n ***{ROIName} Fit Results***\n\n'+ ' Equation: '+self.fitLineEqn + FitStats1 + FitStats2 + refs
    
        #generate new window to plot the results

        if generateSeperateWindows:
            self.windowNames[ROIName] =  MaskedScatterPlotFit([self.xData,self.yData],[self.xData,self.yyData],img_selected,
                                                        self.maskedImage,fitStats, self.fitLineEqn, self.nameTuple)
            self.windowNames[ROIName].show()


        '''  
        from scipy.linalg import lstsq
        M = xData[:, np.newaxis]**[0, 1] #use >1 for polynomial fits
        p, res, rnk, s = lstsq(M, yData)
        yyData = p[0] + p[1]*xData
        '''
    
    def updateROIDict(self):
        self.rois = {
                    'ROI 1':(self.scatter_mask,self.rb_roi1.isChecked()),
                    'ROI 2':(self.scatter_mask2,self.rb_roi2.isChecked()),
                    'ROI 3':(self.scatter_mask3,self.rb_roi3.isChecked())
                    }

    def applyMultipleROIs(self):
        with pg.BusyCursor():
            self.updateROIDict()
            for key in self.rois.keys():
                if self.rois[key][1]:
                    self.getMaskRegion(key)
                else:
                    pass

    def addMultipleROIs(self):
        self.updateROIDict()
        for key in self.rois.keys():
            if self.rois[key][1]:
                self.createMask(self.rois[key][0])
            else:
                self.clearMask(self.rois[key][0])

    def clearMultipleROIs(self):
        self.updateROIDict()
        for key in self.rois.keys():
            if not self.rois[key][1]:
                self.clearMask(self.rois[key][0])
            else:
                pass

    def createCompositeScatter(self):
        
        points = []
        fitLine = []
        roiFitEqn = {}

        self.updateROIDict()
        for n, key in enumerate(self.rois.keys()):
            if self.rois[key][1]:
                self.getMaskRegion(key, generateSeperateWindows = False)
                points.append(np.column_stack([self.xData,self.yData]))
                fitLine.append(np.column_stack([self.xData,self.yyData]))
                roiFitEqn[key] = self.fitLineEqn
            else:
                pass
        
        logger.info(f' fitline shape: {np.shape(fitLine)}')
        logger.info(f' points shape: {np.shape(points)}')
        self.compositeScatterWindow = CompositeScatterPlot(np.array(points),np.array(fitLine),roiFitEqn, self.nameTuple)
        self.compositeScatterWindow.show()

    def _createCompositeScatter(self):
        self.scatterColors = ['w', 'c', 'y', 'k', 'm']
        points = []
        fitLine = []

        self.updateROIDict()
        for n, key in enumerate(self.rois.keys()):
            if self.rois[key][1]:
                self.getMaskRegion(key, generateSeperateWindows = False)
        
                for x,y,yy in zip(self.xData, self.yData,self.yyData):
                    
                    points.append({'pos': (x,y), 'data': 'id', 'size': 3, 'pen': pg.mkPen(None),
                                    'brush': self.scatterColors[n]})
                fitLine.extend(np.column_stack((self.xData,self.yyData)))
            else:
                pass

        logger.info(f' fitline shape: {np.shape(fitLine)}')
        self.compositeScatterWindow = CompositeScatterPlot(points,np.array(fitLine))
        self.compositeScatterWindow.show()                     

    def getROIParams(self):
        print(np.array(self.scatter_mask.getSceneHandlePositions()))

class MaskedScatterPlotFit(QtWidgets.QMainWindow):

    def __init__(self, scatterData, fitData, mask, maskedImage,fitString, fitEquation, nameTuple):
        super(MaskedScatterPlotFit, self).__init__()

        uic.loadUi(os.path.join(ui_path, 'uis/maskedScatterPlotFit.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())
        self.scatterData = scatterData
        self.fitData = fitData
        self.mask = mask
        self.maskedImage = maskedImage
        self.fitString = fitString
        self.fitEquation = fitEquation
        self.nameTuple = nameTuple
       
        #set the graphicslayoutwidget in the ui as canvas
        self.canvas = self.scatterViewer.addPlot()
        self.canvas.addLegend()
        self.canvas.setLabel('bottom', self.nameTuple[0], 'counts')
        self.canvas.setLabel('left', self.nameTuple[1], 'counts')
        self.gb_maskedImage1.setTitle(f" Masked {self.nameTuple[0]}")
        
        #generate a scatter plot item
        self.scattered = pg.ScatterPlotItem(size=3.5, pen=pg.mkPen(None), brush=pg.mkBrush(5, 214, 255, 200))

        #set scatter plot data
        self.scattered.setData(scatterData[0],scatterData[1],name='Data')

        #set z value negative to show scatter data behind the fit line
        self.scattered.setZValue(-10)

        #add scatter plot to the canvas
        self.canvas.addItem(self.scattered)

        #generate plotitem for fit line 
        self.fitLinePlot = pg.PlotDataItem(pen=pg.mkPen(pg.mkColor(220,20,60), width=3.3))

        #set line plot data
        self.fitLinePlot.setData(fitData[0],fitData[1], name = 'Linear Fit')

        #add line plot to the canvas
        self.canvas.addItem(self.fitLinePlot)

        #display Mask
        self.imageView_mask.setImage(self.mask)
        self.imageView_mask.ui.menuBtn.hide()
        self.imageView_mask.ui.roiBtn.hide()
        self.imageView_mask.setPredefinedGradient('plasma')
        
        #display masked Image
        self.imageView_maskedImage.setImage(self.maskedImage)
        self.imageView_maskedImage.ui.menuBtn.hide()
        self.imageView_maskedImage.ui.roiBtn.hide()
        self.imageView_maskedImage.setPredefinedGradient('viridis')

        #display Fit stats
        self.text_fit_results.setPlainText(fitString)
        self.canvas.setTitle(self.fitEquation, color = 'r')

        #connections
        self.pb_copy_results.clicked.connect(self.copyFitResults)
        self.pb_save_results.clicked.connect(self.saveFitResults)
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSaveMask.triggered.connect(self.saveMask)
        self.actionSaveMaskedImage.triggered.connect(self.saveImage)

    def saveFitResults(self):
        S__File = QFileDialog.getSaveFileName(self, "save txt", 'correlationPlotFit.txt', 'txt data (*txt)')

        Text = self.text_fit_results.toPlainText()
        if S__File[0]:
            with open(S__File[0], 'w') as file:
                file.write(Text)

    def copyFitResults(self):
        self.text_fit_results.selectAll()
        self.text_fit_results.copy()
        self.statusbar.showMessage(f"text copied to clipboard")

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.canvas)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save correlation", 'scatterData.csv', 'spectrum and fit (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]))
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            pass

    def saveImage(self):

        file_name = QFileDialog().getSaveFileName(self, "Save image data", 'image.tiff', 'image file(*tiff *tif )')
        if file_name[0]:
            tf.imsave(str(file_name[0]), self.maskedImage)
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            self.statusbar.showMessage('Saving cancelled')
            pass

    def saveMask(self):

        file_name = QFileDialog().getSaveFileName(self, "Save image data", 'mask.tiff', 'image file(*tiff *tif )')
        if file_name[0]:
            tf.imsave(str(file_name[0]), self.mask)
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            self.statusbar.showMessage('Saving cancelled')
            pass

class ComponentScatterPlot(QtWidgets.QMainWindow):

    def __init__(self, decomp_stack, specs):
        super(ComponentScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_path, 'uis/ComponentScatterPlot.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())
        self.w1 = self.scatterViewer.addPlot()
        self.decomp_stack = decomp_stack
        self.specs = specs
        (self.dim1, self.dim3, self.dim2) = self.decomp_stack.shape
        # fill the combonbox depending in the number of components for scatter plot
        for n, combs in enumerate(combinations(np.arange(self.dim1), 2)):
            self.cb_scatter_comp.addItem(str(combs))
            self.cb_scatter_comp.setItemData(n, combs)

        self.s1 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))

        # connections
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSave_Images.triggered.connect(self.tiff_export_images)
        self.pb_updateComponents.clicked.connect(self.setImageAndScatterPlot)
        self.pb_define_mask.clicked.connect(self.createMask)
        self.pb_apply_mask.clicked.connect(self.getMaskRegion)
        self.pb_reset_mask.clicked.connect(self.resetMask)
        self.pb_addALine.clicked.connect(lambda: self.createMask(Line=True))

    def setImageAndScatterPlot(self):

        try:
            self.s1.clear()
        except:
            pass

        comp_tuple = self.cb_scatter_comp.currentData()
        self.img1,self.img2  = self.decomp_stack[comp_tuple[0]],self.decomp_stack[comp_tuple[-1]]
        self.image_view.setImage(self.decomp_stack[comp_tuple[0]])
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient('bipolar')

        self.image_view2.setImage(self.decomp_stack[comp_tuple[-1]])
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()
        self.image_view2.setPredefinedGradient('bipolar')

        points = []
        for i,j in zip(self.img1.flatten(),self.img2.flatten()):

                points.append({'pos': (i,j), 'data' : 'id', 'size': 5, 'pen': pg.mkPen(None),
                           'brush': pg.mkBrush(255, 255, 0, 160)})

        self.s1.addPoints(points)
        self.w1.addItem(self.s1)
        #self.s1.setData(self.specs[:, comp_tuple[0]], self.specs[:, comp_tuple[-1]])
        self.w1.setLabel('bottom', f'PC{comp_tuple[0]+1}')
        self.w1.setLabel('left', f'PC{comp_tuple[-1]+1}')
        self.label_im1.setText(f'PC{comp_tuple[0]+1}')
        self.label_im2.setText(f'PC{comp_tuple[-1]+1}')

    def createMask(self, Line = False):

        self.size = self.img1.max() / 10
        self.pos = int(self.img1.mean())

        if Line:
            self.lineROI = pg.LineSegmentROI([0, 1],pos=(self.pos, self.pos),
                                             pen=pg.mkPen('r', width=4),hoverPen=pg.mkPen('g', width=4),
                                             removable=True)
            self.w1.addItem(self.lineROI)

        else:

            self.scatter_mask = pg.PolyLineROI([[0, 0], [0, self.size], [self.size, self.size], [self.size, 0]],
                                               pos=(self.pos, self.pos), pen=pg.mkPen('r', width=4),
                                               hoverPen=pg.mkPen('g', width=4),
                                               closed=True, removable=True)

            self.w1.addItem(self.scatter_mask)

    def clearMask(self):
        try:
            self.w1.removeItem(self.scatter_mask)
        except AttributeError:
            pass

    def clearPgPlot(self):
        try:
            self.masked_img.close()
        except:
            pass

    def getMaskRegion(self):

        # Ref : https://stackoverflow.com/questions/57719303/how-to-map-mouse-position-on-a-scatterplot

        roiShape = self.scatter_mask.mapToItem(self.s1, self.scatter_mask.shape())
        self._points = list()
        logger.info("Building Scatter Plot Window; Please wait..")
        for i in range(len(self.img1.flatten())):
            self._points.append(QtCore.QPointF(self.img1.flatten()[i], self.img2.flatten()[i]))

        selected = [roiShape.contains(pt) for pt in self._points]
        img_selected = np.reshape(selected, (self.img1.shape))

        self.masked_img = singleStackViewer(img_selected * self.img1, gradient='bipolar')
        self.masked_img.show()

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.w1)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save correlation", '', 'spectrum and fit (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]) + '.csv')
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            pass

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, "save images", '', 'spectrum and fit (*tiff)')
        if file_name[0]:
            tf.imsave(str(file_name[0]) + '.tiff', np.dstack([self.img1, self.img2]).T)
            self.statusbar.showMessage(f"Images saved to {str(file_name[0])}")
        else:
            pass

class LoadingScreen(QtWidgets.QSplashScreen):

    def __init__(self):
        super(LoadingScreen, self).__init__()
        uic.loadUi(os.path.join(ui_path, 'uis/animationWindow.ui'), self)
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

class CompositeScatterPlot(QtWidgets.QMainWindow):

    def __init__(self, scatterPoints,fitLine, fitEquations, nameTuple):
        super(CompositeScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_path, 'uis/multipleScatterFit.ui'), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_path,'css/defaultStyle.css')).read())

        self.scatterPoints = scatterPoints
        self.fitLine = fitLine
        self.scatterColors = ['r', 'g', 'c', 'w', 'k']
        self.fitColors = ['b', 'r', 'w', 'k', 'b']
        self.roiNames = list(fitEquations.keys())
        self.fitEqns = list(fitEquations.values())
        self.nameTuple = nameTuple

        #self.scatterViewer.setBackground('w')
        #set the graphicslayoutwidget in the ui as canvas
        self.canvas = self.scatterViewer.addPlot()
        self.canvas.addLegend()
        self.canvas.setLabel('bottom', self.nameTuple[0], 'counts')
        self.canvas.setLabel('left', self.nameTuple[1], 'counts')

        #connections
        self.actionExport.triggered.connect(self.exportData)
        self.actionSave_as_PNG.triggered.connect(self.exportAsPNG)

        with pg.BusyCursor():
        
            for arr, fitline, clr, fitClr, rname, feqn in zip(self.scatterPoints,self.fitLine,
                                                self.scatterColors,self.fitColors,self.roiNames,self.fitEqns):

                sctrPoints = []
                for pt in arr:
                    sctrPoints.append({'pos': (pt[0],pt[1]), 'data': 'id', 'size': 3, 
                                        'pen': pg.mkPen(None),'brush': clr})

                #generate a scatter plot item
                self.scattered = pg.ScatterPlotItem(size=3.5, pen=clr, brush=pg.mkBrush(5, 214, 255, 200))
                #set scatter plot data
                self.scattered.setPoints(sctrPoints, name  = rname)

                #set z value negative to show scatter data behind the fit line
                self.scattered.setZValue(-10)

                #add scatter plot to the canvas
                self.canvas.addItem(self.scattered)

                
                #generate plotitem for fit line 
                self.fitLinePlot = pg.PlotDataItem(pen=pg.mkPen(fitClr, width=4.5))

                #set line plot data
                self.fitLinePlot.setData(fitline, name = feqn)

                #add line plot to the canvas
                self.canvas.addItem(self.fitLinePlot)

    def exportData(self):

        exporter = pg.exporters.CSVExporter(self.canvas)
        #exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "Save CSV Data", 'scatter.csv', 'image file (*csv)')
        if file_name[0]:
            exporter.export(str(file_name[0]))
            self.statusbar.showMessage(f"Data saved to {str(file_name[0])}")
        else:
            pass

    def exportAsPNG(self):
        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save Image", 'image.png',
                                                            'PNG(*.png);; TIFF(*.tiff);; JPG(*.jpg)')
        exporter = pg.exporters.ImageExporter(self.canvas)

        if file_name[0]:
            exporter.export(str(file_name[0]))
            self.statusbar.showMessage(f"Image saved to {str(file_name[0])}")
        else:
            pass

class MaskSpecViewer(QtWidgets.QMainWindow):

    def __init__(self, xanes_stack=None, xrf_map=None, energy=[]):
        super(MaskSpecViewer, self).__init__()
        uic.loadUi(os.path.join(ui_path,'uis/MaskedView.ui'), self)

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
        self.xrf_view.setPredefinedGradient('viridis')

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

""" Helper Functions"""

def get_xrf_data(h='h5file'):
    """
    get xrf stack from h5 data generated at NSLS-II beamlines

     Arguments:
        h5/hdf5 file

     Returns:
         norm_xrf_stack -  xrf stack image normalized with Io
         mono_e  - excitation enegy used for xrf
         beamline - identity of the beamline
         Io_avg - an average Io value, used before taking log

    """

    f = h5py.File(h, 'r')

    if list(f.keys())[0] == 'xrfmap':
        logger.info('Data from HXN/TES/SRX')
        beamline = f['xrfmap/scan_metadata'].attrs['scan_instrument_id']

        try:

            beamline_scalar = {'HXN': 2, 'SRX': 0, 'TES': 0}

            if beamline in beamline_scalar.keys():

                Io = np.array(f['xrfmap/scalers/val'])[:, :, beamline_scalar[beamline]]
                raw_xrf_stack = np.array(f['xrfmap/detsum/counts'])
                norm_xrf_stack = raw_xrf_stack
                Io_avg = int(remove_nan_inf(Io).mean())
            else:
                logger.error('Unknown Beamline Scalar')
        except:
            logger.warning('Unknown Scalar: Raw Detector count in use')
            norm_xrf_stack = np.array(f['xrfmap/detsum/counts'])

    elif list(f.keys())[0] == 'xrmmap':
        logger.info('Data from XFM')
        beamline = 'XFM'
        raw_xrf_stack = np.array(f['xrmmap/mcasum/counts'])
        Io = np.array(f['xrmmap/scalars/I0'])
        norm_xrf_stack = raw_xrf_stack 
        Io_avg = int(remove_nan_inf(Io).mean())

    else:
        logger.error('Unknown Data Format')

    try:
        mono_e = int(f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy'] * 1000)
        logger.info("Excitation energy was taken from the h5 data")

    except:
        mono_e = 12000
        logger.info(f'Unable to get Excitation energy from the h5 data; using default value {mono_e} KeV')

    return remove_nan_inf(norm_xrf_stack.transpose((2,0,1))), mono_e + 1500, beamline, Io_avg

def remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im

def rebin_image(im, bin_factor):
    arrx, arry = np.shape(im)
    if arrx / bin_factor != int or arrx / bin_factor != int:
        logger.error('Invalid Binning')

    else:
        shape = (arrx / bin_factor, arry / bin_factor)
        return im.reshape(shape).mean(-1).mean(1)

def remove_hot_pixels(image_array, NSigma=5):
    image_array = remove_nan_inf(image_array)
    a, b, c = np.shape(image_array)
    img_stack2 = np.zeros((a, b, c))
    for i in range(a):
        im = image_array[i, :, :]
        im[abs(im) > np.std(im) * NSigma] = im.mean()
        img_stack2[i, :, :] = im
    return img_stack2

def smoothen(image_array, w_size=5):
    a, b, c = np.shape(image_array)
    spec2D_Matrix = np.reshape(image_array, (a, (b * c)))
    smooth2D_Matrix = savgol_filter(spec2D_Matrix,  w_size, w_size - 2, axis = 0)
    return remove_nan_inf(np.reshape(smooth2D_Matrix, (a, b, c)))

def resize_stack(image_array, upscaling = False, scaling_factor = 2):
    en, im1, im2 = np.shape(image_array)

    if upscaling:
        im1_ = im1 * scaling_factor
        im2_ = im2 * scaling_factor
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    else:
        im1_ = int(im1/scaling_factor)
        im2_ = int(im2/scaling_factor)
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    return img_stack_resized

def normalize(image_array, norm_point=-1):
    norm_stack = image_array/image_array[norm_point]
    return remove_nan_inf(norm_stack)

def remove_edges(image_array):
    # z, x, y = np.shape(image_array)
    return image_array[:, 1:- 1, 1:- 1]

def background_value(image_array):
    img = image_array.mean(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg

def background_subtraction(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    ref_image = np.reshape(img_stack.mean(0), (b * c))
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bg_ = np.max(sorted(ref_image)[0:bg_ratio])
    bged_img_stack = img_stack - bg_[:, np.newaxis, np.newaxis]
    return bged_img_stack

def background_subtraction2(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bged_img_stack = img_stack.copy()

    for n, img in enumerate(img_stack):
        bg_ = np.max(sorted(img.flatten())[0:bg_ratio])
        print(bg_)
        bged_img_stack[n] = img - bg_

    return remove_nan_inf(bged_img_stack)

def background1(img_stack):
    img = img_stack.sum(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg

def get_sum_spectra(image_array):
    spec = np.sum(image_array, axis=(1,2))
    return spec

def get_mean_spectra(image_array):
    spec = np.mean(image_array, axis=(1,2))
    return spec

def flatten_(image_array):
    z, x, y = np.shape(image_array)
    flat_array = np.reshape(image_array, (x * y, z))
    return flat_array

def image_to_pandas(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, (a, (b * c)))
    a, b = im_array.shape
    df = pd.DataFrame(data=im_array[:, :],
                      index=['e' + str(i) for i in range(a)],
                      columns=['s' + str(i) for i in range(b)])
    return df

def neg_log(image_array):
    absorb = -1 * np.log(image_array)
    return remove_nan_inf(absorb)

def clean_stack(img_stack, auto_bg=False, bg_percentage=5):
    a, b, c = np.shape(img_stack)

    if auto_bg == True:
        bg_ = background1(img_stack)

    else:
        sum_spec = (img_stack.sum(1)).sum(1)
        ref_stk_num = np.where(sum_spec == sum_spec.max())[-1]

        ref_image = np.reshape(img_stack[ref_stk_num], (b * c))
        bg_ratio = int((b * c) * 0.01 * bg_percentage)
        bg_ = np.max(sorted(ref_image)[0:bg_ratio])

    bg = np.where(img_stack[ref_stk_num] > bg_, img_stack[ref_stk_num], 0)
    bg2 = np.where(bg < bg_, bg, 1)

    bged_img_stack = img_stack * bg2

    return remove_nan_inf(bged_img_stack)

def subtractBackground(im_stack,bg_region):
    if bg_region.ndim == 3:
        bg_region_ = np.mean(bg_region, axis = (1,2))

    elif bg_region.ndim == 2:
        bg_region_ = np.mean(bg_region, axis = 1)

    else: bg_region_ = bg_region

    return im_stack - bg_region_[:, np.newaxis, np.newaxis]

def classify(img_stack, correlation='Pearson'):
    img_stack_ = img_stack
    a, b, c = np.shape(img_stack_)
    norm_img_stack = normalize(img_stack_)
    f = np.reshape(norm_img_stack, (a, (b * c)))

    max_x, max_y = np.where(norm_img_stack.sum(0) == (norm_img_stack.sum(0)).max())
    ref = norm_img_stack[:, int(max_x), int(max_y)]
    corr = np.zeros(len(f.T))
    for s in range(len(f.T)):
        if correlation == 'Kendall':
            r, p = stats.kendalltau(ref, f.T[s])
        elif correlation == 'Pearson':
            r, p = stats.pearsonr(ref, f.T[s])

        corr[s] = r

    cluster_image = np.reshape(corr, (b, c))
    return (cluster_image ** 3), img_stack_

def correlation_kmeans(img_stack, n_clusters, correlation='Pearson'):
    img, bg_image = classify(img_stack, correlation)
    img[np.isnan(img)] = -99999
    X = img.reshape((-1, 1))
    k_means = sc.KMeans(n_clusters)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape) + 1

    return X_cluster

def cluster_stack(im_array, method='KMeans', n_clusters_=4, decomposed=False, decompose_method='PCA',
                  decompose_comp=2):
    a, b, c = im_array.shape

    if method == 'Correlation-Kmeans':

        X_cluster = correlation_kmeans(im_array, n_clusters_, correlation='Pearson')

    else:

        methods = {'MiniBatchKMeans': sc.MiniBatchKMeans, 'KMeans': sc.KMeans,
                   'MeanShift': sc.MeanShift, 'Spectral Clustering': sc.SpectralClustering,
                   'Affinity Propagation': sc.AffinityPropagation}

        if decomposed:
            im_array = denoise_with_decomposition(im_array, method_=decompose_method,
                                                  n_components=decompose_comp)

        flat_array = np.reshape(im_array, (a, (b * c)))
        init_cluster = methods[method](n_clusters=n_clusters_)
        init_cluster.fit(np.transpose(flat_array))
        X_cluster = init_cluster.labels_.reshape(b, c) + 1

    decon_spectra = np.zeros((a, n_clusters_))
    decon_images = np.zeros((n_clusters_, b, c))

    for i in range(n_clusters_):
        mask_i = np.where(X_cluster == (i + 1), X_cluster, 0)
        spec_i = get_sum_spectra(im_array * mask_i)
        decon_spectra[:, i] = spec_i
        decon_images[i] = im_array.sum(0) * mask_i

    return decon_images, X_cluster, decon_spectra

def kmeans_variance(im_array):
    a, b, c = im_array.shape
    flat_array = np.reshape(im_array, (a, (b * c)))
    var = np.arange(24)
    clust_n = np.arange(24) + 2

    for clust in var:
        init_cluster = sc.KMeans(n_clusters=int(clust + 2))
        init_cluster.fit(np.transpose(flat_array))
        var_ = init_cluster.inertia_
        var[clust] = np.float64(var_)

    kmeans_var_plot = pg.plot(clust_n, var , title = 'KMeans Variance',
                              pen = pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')
    kmeans_var_plot.setLabel('bottom', 'Cluster Number')
    kmeans_var_plot.setLabel('left', 'Sum of squared distances')

def pca_scree(im_stack):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    pca = sd.PCA(z)
    pca.fit(img_)
    #var = pca.explained_variance_ratio_
    var = pca.singular_values_

    pca_scree_plot = pg.plot(var[:24], title = 'PCA Scree Plot',
                              pen = pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')
    pca_scree_plot.addLine(y=0)
    pca_scree_plot.setLabel('bottom', 'Component Number')
    pca_scree_plot.setLabel('left', 'Singular Values')

def decompose_stack(im_stack, decompose_method='PCA', n_components_=3):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    methods_dict = {'PCA': sd.PCA, 'IncrementalPCA': sd.IncrementalPCA,
                    'NMF': sd.NMF, 'FastICA': sd.FastICA, 'DictionaryLearning': sd.MiniBatchDictionaryLearning,
                    'FactorAnalysis': sd.FactorAnalysis, 'TruncatedSVD': sd.TruncatedSVD}

    _mdl = methods_dict[decompose_method](n_components=n_components_)

    ims = (_mdl.fit_transform(img_).reshape(x, y, n_components_)).transpose(2, 1, 0)
    spcs = _mdl.components_.transpose()
    decon_spetra = np.zeros((z, n_components_))
    decom_map = np.zeros((ims.shape))

    for i in range(n_components_):
        f = ims.copy()[i]
        f[f < 0] = 0
        spec_i = ((new_image.T * f).sum(1)).sum(1)
        decon_spetra[:, i] = spec_i

        f[f > 0] = i + 1
        decom_map[i] = f
    decom_map = decom_map.sum(0)

    return np.float32(ims), spcs, decon_spetra, decom_map

def denoise_with_decomposition(img_stack, method_='PCA', n_components=4):
    new_image = img_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))

    methods_dict = {'PCA': sd.PCA, 'IncrementalPCA': sd.IncrementalPCA,
                    'NMF': sd.NMF, 'FastICA': sd.FastICA, 'DictionaryLearning': sd.DictionaryLearning,
                    'FactorAnalysis': sd.FactorAnalysis, 'TruncatedSVD': sd.TruncatedSVD}

    decomposed = methods_dict[method_](n_components=n_components)

    ims = (decomposed.fit_transform(img_).reshape(x, y, n_components)).transpose(2, 1, 0)
    ims[ims < 0] = 0
    ims[ims > 0] = 1
    mask = ims.sum(0)
    mask[mask > 1] = 1
    # mask = uniform_filter(mask)
    filtered = img_stack * mask
    # plt.figure()
    # plt.imshow(filtered.sum(0))
    # plt.title('background removed')
    # plt.show()
    return remove_nan_inf(filtered)

def interploate_E(refs, e):
    n = np.shape(refs)[1]
    refs = np.array(refs)
    ref_e = refs[:, 0]
    ref = refs[:, 1:n]
    all_ref = []
    for i in range(n - 1):
        ref_i = np.interp(e, ref_e, ref[:, i])
        all_ref.append(ref_i)
    return np.array(all_ref)

def getStats(spec,fit, num_refs = 2):
    stats = {}

    r_factor = (np.sum(spec -fit)**2) / np.sum(spec**2)
    stats['R_Factor'] = np.around(r_factor,5)

    y_mean = np.sum(spec)/len(spec)
    SS_tot = np.sum((spec-y_mean)**2)
    SS_res = np.sum((spec - fit)**2)
    r_square = 1 - (SS_res/SS_tot)
    stats['R_Square'] = np.around(r_square,4)

    chisq = np.sum((spec - fit) ** 2)
    stats['Chi_Square'] = np.around(chisq,5)

    red_chisq = chisq/(len(spec) - num_refs)
    stats['Reduced Chi_Square'] = red_chisq

    return stats

def xanes_fitting_1D(spec, e_list, refs, method='NNLS', alphaForLM = 0.01):
    """Linear combination fit of image data with reference standards"""

    int_refs = (interploate_E(refs, e_list))

    if method == 'NNLS':
        coeffs, r = opt.nnls(int_refs.T, spec)

    elif method == 'LASSO':
        lasso = linear_model.Lasso(positive=True, alpha=alphaForLM) #lowering alpha helps with 1D fits
        fit_results = lasso.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == 'RIDGE':
        ridge = linear_model.Ridge(alpha=alphaForLM)
        fit_results = ridge.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    fit = coeffs@int_refs
    stats = getStats(spec,fit, num_refs = np.min(np.shape(int_refs.T)))

    return stats, coeffs

def xanes_fitting(im_stack, e_list, refs, method='NNLS',alphaForLM = 0.1, binStack = False):
    """Linear combination fit of image data with reference standards"""
    
    if binStack:
        im_stack = resize_stack(im_stack, scaling_factor = 4)

    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    r_factor_arr = []
    lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)
    for n, i in enumerate(range(im1 * im2)):
        stats, coeffs = xanes_fitting_1D(im_array[:, i], e_list, refs, method=method, alphaForLM=alphaForLM)
        coeffs_arr.append(coeffs)
        r_factor_arr.append(stats['R_Factor'])

    abundance_map = np.reshape(coeffs_arr, (im1, im2, -1))
    r_factor_im = np.reshape(r_factor_arr, (im1, im2))

    return abundance_map, r_factor_im, np.mean(coeffs_arr,axis=0)

def xanes_fitting_Line(im_stack, e_list, refs, method='NNLS',alphaForLM = 0.05):
    """Linear combination fit of image data with reference standards"""
    en, im1, im2 = np.shape(im_stack)
    im_array = np.mean(im_stack,2)
    coeffs_arr = []
    meanStats = {'R_Factor':0,'R_Square':0,'Chi_Square':0,'Reduced Chi_Square':0}

    for i in range(im1):
        stats, coeffs = xanes_fitting_1D(im_array[:, i], e_list, refs,
                                         method=method, alphaForLM=alphaForLM)
        coeffs_arr.append(coeffs)
        for key in stats.keys():
            meanStats[key] += stats[key]

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals/im1),5)

    return meanStats, np.mean(coeffs_arr,axis=0)

def xanes_fitting_Binned(im_stack, e_list, refs, method='NNLS',alphaForLM = 0.05):
    """Linear combination fit of image data with reference standards"""
    
    im_stack = resize_stack(im_stack, scaling_factor = 10)
    #use a simple filter to find threshold value
    val = filters.threshold_otsu(im_stack[-1])
    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    meanStats = {'R_Factor':0,'R_Square':0,'Chi_Square':0,'Reduced Chi_Square':0}

    specs_fitted = 0
    total_spec = im1*im2
    for i in range(total_spec):
        spec = im_array[:, i]
        #do not fit low intensity/background regions
        if spec[-1]>val:
            specs_fitted +=1
            stats, coeffs = xanes_fitting_1D(spec/spec[-1], e_list, refs,
                                            method=method, alphaForLM=alphaForLM)
            coeffs_arr.append(coeffs)
            for key in stats.keys():
                meanStats[key] += stats[key]
        else:
            pass

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals/specs_fitted),6)
    #print(f"{specs_fitted}/{total_spec}")
    return meanStats, np.mean(coeffs_arr,axis=0)

def create_df_from_nor(athenafile='fe_refs.nor'):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    skip_raw_n = n_refs+6

    df = pd.read_table(athenafile, delim_whitespace=True, skiprows=skip_raw_n,
                       header=None, usecols=np.arange(0, n_refs))
    df2 = pd.read_table(athenafile, delim_whitespace=True, skiprows=skip_raw_n-1,
                        usecols=np.arange(0, n_refs + 1))
    new_col = df2.columns.drop('#')
    df.columns = new_col
    return df, list(new_col)

def create_df_from_nor_try2(athenafile='fe_refs.nor'):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    df_refs = pd.DataFrame(refs)

    df = pd.read_csv(athenafile, header=None)
    new_col = list((str(df.iloc[n_refs + 5].values)).split(' ')[2::2])
    df_refs.columns = new_col

    return df_refs, list(new_col)

def energy_from_logfile(logfile = 'maps_log_tiff.txt'):
    df = pd.read_csv(logfile, header= None, delim_whitespace=True, skiprows=9)
    return df[9][df[7]=='energy'].values.astype(float)

def xanesNormalization(e, mu, e0=7125, step=None,
            nnorm=2, nvict=0, pre1=None, pre2=-50,
            norm1=100, norm2=None, guess = False):
    if guess:
        result = preedge(e, mu, e0, step = step, nnorm=nnorm,
                         nvict = nvict)

        return result['pre1'],result['pre2'],result['norm1'],result['norm2']

    else:
        result = preedge(e, mu, e0, step, nnorm,
                         nvict, pre1, pre2, norm1, norm2)

        return result['pre_edge'],result['post_edge'],result['norm']

def xanesNormStack(e_list,im_stack, e0=7125, step=None,
            nnorm=2, nvict=0, pre1=None, pre2=-50,
            norm1=100, norm2=None):

    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    normedStackArray = np.zeros_like(im_array)

    for i in range(im1 * im2):
        pre_line, post_line, normXANES = xanesNormalization(e_list, im_array[:, i], e0=e0, step=step,
                           nnorm=nnorm, nvict=nvict, pre1=pre1, pre2=pre2,
                           norm1=norm1, norm2=norm2, guess=False)
        normedStackArray[:, i] = normXANES

    return remove_nan_inf(np.reshape(normedStackArray,(en, im1, im2)))

def align_stack(stack_img, ref_image_void = True, ref_stack = None, transformation = StackReg.TRANSLATION,
                reference = 'previous'):

    ''' Image registration flow using pystack reg'''

    # all the options are in one function

    sr = StackReg(transformation)

    if ref_image_void:
        tmats_ = sr.register_stack(stack_img, reference=reference)

    else:
        tmats_ = sr.register_stack(ref_stack, reference=reference)
        out_ref = sr.transform_stack(ref_stack)

    out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk), tmats_

def align_simple(stack_img, transformation = StackReg.TRANSLATION, reference = 'previous'):

    sr = StackReg(transformation)
    tmats_ = sr.register_stack(stack_img, reference = 'previous')
    for i in range(10):
        out_stk = sr.transform_stack(stack_img, tmats=tmats_)
        import time
        time.sleep(2)
    return np.float32(out_stk)

def align_with_tmat(stack_img, tmat_file, transformation = StackReg.TRANSLATION):

    sr = StackReg(transformation)
    out_stk = sr.transform_stack(stack_img, tmats=tmat_file)
    return np.float32(out_stk)

def align_stack_iter(stack, ref_stack_void = True, ref_stack = None, transformation = StackReg.TRANSLATION,
                     method=('previous', 'first'), max_iter=2):
    if  ref_stack_void:
        ref_stack = stack

    for i in range(max_iter):
        sr = StackReg(transformation)
        for ii in range(len(method)):
            print(ii,method[ii])
            tmats = sr.register_stack(ref_stack, reference=method[ii])
            ref_stack = sr.transform_stack(ref_stack)
            stack = sr.transform_stack(stack, tmats=tmats)

    return np.float32(stack)

def modifyStack(raw_stack, normalizeStack = False, normToPoint = -1,
                applySmooth = False, smoothWindowSize = 3,
                applyThreshold = False, thresholdValue = 0,
                removeOutliers = False, nSigmaOutlier = 3,
                applyTranspose = False, transposeVals = (0,1,2),
                applyCrop = False, cropVals = (0,1,2), removeEdges = False,
                resizeStack = False, upScaling = False, binFactor = 2
                ):


    ''' A giant function to modify the stack with many possible operations.
        all the changes can be saved to a jason file as a config file. Enabling and
        distabling the sliders is a problem'''

    '''
    normStack = normalize(raw_stack, norm_point=normToPoint)
    smoothStack = smoothen(raw_stack, w_size= smoothWindowSize)
    thresholdStack = clean_stack(raw_stack, auto_bg=False, bg_percentage = thresholdValue)
    outlierStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)
    transposeStack = np.transpose(raw_stack, transposeVals)
    croppedStack = raw_stack[cropVals]
    edgeStack = remove_edges(raw_stack)
    binnedStack = resize_stack(raw_stack,upscaling=upScaling,scaling_factor=binFactor)
    
    '''

    if removeOutliers:
        modStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)

    else:
        modStack = raw_stack

    if applyThreshold:
        modStack = clean_stack(modStack, auto_bg=False, bg_percentage=thresholdValue)

    else:
        pass

    if applySmooth:
        modStack = smoothen(modStack, w_size=smoothWindowSize)

    else:
        pass

    if applyTranspose:
        modStack = np.transpose(modStack, transposeVals)

    else:
        pass

    if applyCrop:
        modStack = modStack[cropVals]

    else:
        pass

    if normalizeStack:
        modStack = normalize(raw_stack, norm_point=normToPoint)
    else:
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
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = midasWindow()
    window.show()
    sys.exit(app.exec_())
