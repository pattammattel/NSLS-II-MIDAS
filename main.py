# -*- coding: utf-8 -*-

# Author: Ajith Pattammattel
# Date:06-23-2020

import logging
import webbrowser

from subprocess import Popen
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from xrf_xanes_3ID_gui import xrf_3ID
from StackPlot import *

logger = logging.getLogger()


class Ui(QtWidgets.QMainWindow):
    def __init__(self, im_stack=None, energy=None, refs = None):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow_admin.ui', self)
        self.im_stack = im_stack
        self.updated_stack = self.im_stack
        self.energy = energy
        self.refs = refs

        self.actionOpen_Image_Data.triggered.connect(self.browse_file)
        self.actionSave_as.triggered.connect(self.save_stack)
        self.actionExit.triggered.connect(self.close)

        self.actionOpen_PyXRF.triggered.connect(self.open_pyxrf)
        self.actionOpen_Image_J.triggered.connect(self.open_imagej)
        self.actionOpen_TomViz.triggered.connect(self.open_tomviz)
        self.actionOpen_Mantis.triggered.connect(self.open_mantis)
        self.actionOpen_Athena.triggered.connect(self.open_athena)
        self.actionOpen_in_GitHub.triggered.connect(self.open_github_link)

        self.actionOpen_HXN_DB.triggered.connect(self.open_db_tools_3id)

        self.cb_log.stateChanged.connect(self.view_stack)
        self.cb_remove_edges.stateChanged.connect(self.view_stack)
        self.cb_norm.stateChanged.connect(self.view_stack)
        self.cb_smooth.stateChanged.connect(self.view_stack)
        self.cb_remove_outliers.stateChanged.connect(self.view_stack)
        self.cb_remove_bg.stateChanged.connect(self.view_stack)
        self.cb_bg_auto.stateChanged.connect(self.view_stack)
        self.sb_smooth_size.valueChanged.connect(self.view_stack)
        self.sb_tolerence.valueChanged.connect(self.view_stack)
        self.dsb_bg_fraction.valueChanged.connect(self.view_stack)
        self.pb_reset_img.clicked.connect(self.reset_and_load_stack)
        self.pb_crop.clicked.connect(self.crop_to_dim)
        self.pb_crop.clicked.connect(self.view_stack)
        self.pb_ref_xanes.clicked.connect(self.select_ref_file)
        self.pb_elist_xanes.clicked.connect(self.select_elist)

        self.pb_set_spec_roi.clicked.connect(self.set_spec_roi)

        # Analysis
        self.pb_pca_scree.clicked.connect(self.pca_scree_)
        self.pb_calc_components.clicked.connect(self.calc_comp_)
        self.pb_kmeans_elbow.clicked.connect(self.kmeans_elbow)
        self.pb_calc_cluster.clicked.connect(self.clustering_)
        self.pb_xanes_fit.clicked.connect(self.fast_xanes_fitting)
        self.show()

    def open_github_link(self):
        webbrowser.open('https://github.com/pattammattel/NSLS-II-MIDAS')

    def open_db_tools_3id(self):
        self._new_window = xrf_3ID()
        self._new_window.show()
        logger.info('opening new working window for HXN-3ID')

    def select_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(folder_path))
        global dest
        dest = self.le_wd.text()

    def open_pyxrf(self):
        logger.info('opening pyXRF GUI')
        try:
            Popen(['pyxrf'])

        except ModuleNotFoundError:

            logger.error('Not connected to the beamline account or not in pyxrf conda env')

    def open_athena(self):
        logger.info('opening ATHENA')
        try:

            athena_path = r'C:\Users\pattammattel\AppData\Roaming\DemeterPerl\perl\site\bin\dathena.bat'
            Popen([athena_path])

        except FileNotFoundError:

            logger.error('Wrong dathena.bat path')

    def open_imagej(self):
        logger.info('opening ImgaeJ')
        try:
            imagej_path = r'C:\Users\pattammattel\Fiji.app\ImageJ-win64.exe'
            Popen([imagej_path])

        except FileNotFoundError:

            logger.error('Wrong ImageJ.exe path')

    def open_smak(self):
        smak_path = r'C:\Program Files\smak\smak.exe'
        Popen([smak_path])

    def open_mantis(self):

        logger.info('opening  Mantis')
        try:
            mantis_path = r'C:\Users\pattammattel\mantis-2.3.02.amd64.exe'
            Popen([mantis_path])

        except FileNotFoundError:
            logger.error('Wrong Mantis.exe path')

    def open_tomviz(self):

        logger.info('opening TomViz')

        try:
            tomviz_path = r'C:\Program Files\tomviz\bin\tomviz.exe'
            Popen([tomviz_path, self.le_tiff_file.text()])

        except FileNotFoundError:
            logger.error('Wrong tomviz.exe path')

    # XRF Loading

    def browse_file(self):
        filename = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*.hdf *.h5 *tiff *tif )')
        self.file_name = (str(filename[0]))
        try:
            self.reset_and_load_stack()
        except:
            pass

    def load_stack(self):
        logger.info('Loading.. please wait...')

        if self.file_name.endswith('.h5'):
            stack_, mono_e = get_xrf_data(self.file_name)
            self.sb_zrange2.setMaximum(10000)
            self.sb_zrange2.setValue(mono_e/10)
            self.sb_zrange1.setValue(100)

        elif self.file_name.endswith('.tiff') or self.file_name.endswith('.tif'):
            stack_ = tf.imread(self.file_name).transpose(1, 2, 0)
            self.sb_zrange1.setValue(0)
            self.sb_zrange2.setMaximum(10000)
            self.sb_zrange2.setValue(stack_.shape[-1])

        else:
            logger.error('Unknown data format')

        try:

            logger.info(f' loaded stack with {np.shape(stack_)} from the file')
            self.im_stack = stack_.T
            self.init_dimZ = self.im_stack.shape[0]
            self.init_dimX = self.im_stack.shape[1]
            self.init_dimY = self.im_stack.shape[2]
            self.sb_xrange2.setMaximum(5000)
            self.sb_yrange2.setMaximum(5000)
            self.sb_xrange2.setValue(self.init_dimX)
            self.sb_yrange2.setValue(self.init_dimY)
            logger.info(f' Transposed to shape: {np.shape(self.im_stack)}')

        except UnboundLocalError:
            logger.error('No file selected')
            pass

        try:
            self.view_stack()
            self.update_stack_info()

        except:
            logger.error('No image file loaded')
            pass

        logger.info(f'completed image shape {np.shape(self.im_stack)}')

    def reset_and_load_stack(self):
        self.cb_log.setChecked(False)
        self.cb_remove_edges.setChecked(False)
        self.cb_norm.setChecked(False)
        self.cb_smooth.setChecked(False)
        self.cb_remove_outliers.setChecked(False)
        self.cb_remove_bg.setChecked(False)
        self.cb_bg_auto.setChecked(False)
        self.sb_xrange1.setValue(0)
        self.sb_yrange1.setValue(0)
        self.load_stack()

    def update_stack_info(self):
        z, x, y = np.shape(self.updated_stack)
        self.sb_zrange2.setMaximum(z+self.sb_zrange1.value())
        self.sb_xrange2.setValue(x)
        self.sb_xrange2.setMaximum(x)
        self.sb_yrange2.setValue(y)
        self.sb_yrange2.setMaximum(y)
        logger.info('Stack info has been updated')

    def crop_to_dim(self):
        x1, x2 = self.sb_xrange1.value(),self.sb_xrange2.value()
        y1, y2 = self.sb_yrange1.value(), self.sb_yrange2.value()
        z1, z2 = self.sb_zrange1.value(), self.sb_zrange2.value()

        self.updated_stack = remove_nan_inf(self.im_stack[z1:z2, x1:x2, y1:y2])

    def update_stack(self):

        self.crop_to_dim()

        if self.cb_remove_outliers.isChecked():
            self.updated_stack = remove_hot_pixels(self.updated_stack, NSigma=self.sb_tolerence.value())
            logger.info(f'Removing Outliers with NSigma {self.sb_tolerence.value()}')

        if self.cb_remove_edges.isChecked():
            self.updated_stack = remove_edges(self.updated_stack)
            logger.info(f'Removed edges, new shape {self.updated_stack.shape}')
            self.update_stack_info()

        if self.cb_remove_bg.isChecked():
            logger.info('Removing background')
            self.updated_stack = clean_stack(self.updated_stack, auto_bg= self.cb_bg_auto.isChecked(),
                                                         bg_percentage=self.dsb_bg_fraction.value())

        if self.cb_log.isChecked():
            self.updated_stack = remove_nan_inf(np.log(self.updated_stack))
            logger.info('Log Stack is in use')

        if self.cb_smooth.isChecked():
            self.updated_stack = smoothen(self.updated_stack, w_size = self.sb_smooth_size.value() )
            logger.info('Spectrum Smoothening Applied')

        if self.cb_norm.isChecked():
            logger.info('Normalizing spectra')
            self.updated_stack = normalize(self.updated_stack,
                                           norm_point=-1)


        logger.info(f'Updated image is in use')

    def view_stack(self):

        if not self.im_stack.ndim == 3:
            raise ValueError("stack should be an ndarray with ndim == 3")
        else:
            self.update_stack()

        try:
            self.image_view.removeItem(self.image_roi)
        except:
            pass

        (self.dim1, self.dim3, self.dim2) = self.updated_stack.shape
        self.image_view.setImage(self.updated_stack)
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
            scaleSnap=True, translateSnap=True, rotateSnap=True, removable=True
        )

        '''
        cn = int(self.dim2 // 2)
        sz = np.max([int(self.dim2 * 0.15),int(self.dim3 * 0.15)])
        self.image_roi = pg.PolyLineROI([[0,0], [0,sz], [sz,sz], [sz,0]],
                                        pos =(int(self.dim2 // 2), int(self.dim3 // 2)), closed=True)


        #self.image_roi.addScaleHandle([10, 1], [0, 0])
        self.image_roi.addRotateHandle([sz//2, sz//2], [2, 2])

        self.image_view.addItem(self.image_roi)
        self.spec_roi = pg.LinearRegionItem(values=(self.stack_center - self.stack_width,
                                                    self.stack_center + self.stack_width))
        self.spec_roi.setBounds([0, self.dim1])
        self.sb_roi_spec_s.setValue(self.stack_center - self.stack_width)
        self.sb_roi_spec_e.setValue(self.stack_center + self.stack_width)
        self.update_spectrum()
        self.update_image_roi()

        # connections
        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)
        self.sb_roi_spec_s.valueChanged.connect(self.set_spec_roi)
        self.sb_roi_spec_e.valueChanged.connect(self.set_spec_roi)
        # self.pb_play_stack.clicked.connect(self.play_stack)

    def update_region(self):
        region = self.image_roi.getArrayRegion(self.updated_stack,self.image_view.imageItem, axes=(1,2))
        print(region.shape)

    def update_spectrum(self):

        xdata = np.arange(self.sb_zrange1.value(), self.sb_zrange2.value(), 1)
        #ydata = remove_nan_inf(get_sum_spectra(self.updated_stack[:, xmin:xmax,ymin:ymax]))
        ydata = self.image_roi.getArrayRegion(self.updated_stack, self.image_view.imageItem, axes=(1, 2))
        sizex, sizey = ydata.shape[1], ydata.shape[2]
        posx, posy = self.image_roi.pos()
        self.le_roi.setText(str(int(posx))+':' +str(int(posy)))
        self.le_roi_size.setText(str(sizex) +','+ str(sizey))

        self.spectrum_view.plot(xdata, get_sum_spectra(ydata), clear=True)
        self.spectrum_view.addItem(self.spec_roi)

    def update_image_roi(self):
        self.spec_lo, self.spec_hi = self.spec_roi.getRegion()
        self.le_spec_roi.setText(str(int(self.spec_lo)) + ':'+ str(int(self.spec_hi)))
        self.le_spec_roi_size.setText(str(int(self.spec_hi-self.spec_lo)))
        self.image_view.setImage(self.updated_stack[int(self.spec_lo):int(self.spec_hi), :, :].mean(0))

    def set_spec_roi(self):
        self.spec_lo_, self.spec_hi_ = int(self.sb_roi_spec_s.value()), int(self.sb_roi_spec_e.value())
        self.spec_roi.setRegion((self.spec_lo_, self.spec_hi_))
        self.update_image_roi()

    def save_stack(self):
        try:
            self.update_stack()
            file_name = QFileDialog().getSaveFileName(self, "", '', 'image file(*tiff *tif )')
            tf.imsave(str(file_name[0]), self.updated_stack.transpose(0,2,1))
            logger.info(f'Updated Image Saved: {str(file_name[0])}')
        except:
            logger.error('No file to save')
            pass

    def open_xrf_stack_imagej(self):
        tf.imsave(f'{self.le_wd.text()}/tmp_image.tiff', np.float32(self.updated_stack), imagej=True)
        imagej_path = r'C:\Users\pattammattel\Fiji.app\ImageJ-win64.exe'
        Popen([imagej_path, f'{self.le_wd.text()}/tmp_image.tiff'])

    # Component Analysis

    def pca_scree_(self):
        logger.info('Process started..')
        self.update_stack()
        pca_scree(self.updated_stack)
        logger.info('Process complete')

    def calc_comp_(self):

        logger.info('Process started..')

        self.update_stack()
        n_components = self.sb_ncomp.value()
        method_ = self.cb_comp_method.currentText()

        ims, comp_spec, decon_spec, decomp_map = decompose_stack(self.updated_stack,
            decompose_method=method_ , n_components_=n_components)

        self._new_window3 = ComponentViewer(ims, comp_spec, decon_spec,decomp_map)
        self._new_window3.show()

        logger.info('Process complete')

    def kmeans_elbow(self):
        logger.info('Process started..')
        self.update_stack()
        try:
            kmeans_variance(self.updated_stack)
            logger.info('Process complete')
        except OverflowError:
            pass
            logger.error('Overflow Error, values are too long')

    def clustering_(self):

        logger.info('Process started..')
        self.update_stack()
        method_ = self.cb_clust_method.currentText()

        decon_images,X_cluster, decon_spectra = cluster_stack(self.updated_stack, method=method_,
                                                   n_clusters_=self.sb_ncluster.value(),
                                                   decomposed=False,
                                                   decompose_method=self.cb_comp_method.currentText(),
                                                   decompose_comp = self.sb_ncomp.value())

        self._new_window4 = ClusterViewer(decon_images,X_cluster, decon_spectra)
        self._new_window4.show()

        logger.info('Process complete')


    # XANES files

    def select_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open reference file", '', 'text file (*.txt *.nor)')
        try:
            self.refs = np.loadtxt(str(file_name[0]))
            if bool(self.refs.max()):
                self.change_color_on_load(self.pb_ref_xanes)
                plot_xanes_refs(self.refs)

        except OSError:
            logger.error('No file selected')
            pass


    def select_elist(self):
        file_name = QFileDialog().getOpenFileName(self, "Open energy list", '', 'text file (*.txt)')
        try:
            self.energy = np.loadtxt(str(file_name[0]))
            if bool(self.energy.max()) == True:
                self.change_color_on_load(self.pb_elist_xanes)

        except OSError:
            logger.error('No file selected')
            pass

    def change_color_on_load(self, button_name):
        button_name.setStyleSheet("background-color : green")

    def fast_xanes_fitting(self):

        logger.info('Process started..')

        e_list1 = self.energy
        ref1 = self.refs
        self.update_stack()

        xanes_maps = xanes_fitting(self.updated_stack, e_list1,ref1,
                                   method=self.cb_xanes_fitting_method.currentText())
        logger.info('Process complete')

        self._new_window5 = XANESViewer(xanes_maps.T, self.updated_stack, e_list1, ref1)
        self._new_window5.show()



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
    window = Ui()
    window.show()
    sys.exit(app.exec_())
