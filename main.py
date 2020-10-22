# -*- coding: utf-8 -*-

# Author: Ajith Pattammattel
# Date:06-23-2020

import logging
from subprocess import Popen
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from xrf_xanes_3ID_gui import xrf_3ID
from StackCalcs import *
from StackPlot import *

logger = logging.getLogger()


class Ui(QtWidgets.QMainWindow):
    def __init__(self, im_stack=None):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow_v5.ui', self)
        self.im_stack = im_stack
        self.updated_stack = self.im_stack

        self.pb_wd.clicked.connect(self.select_wd)
        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        # self.pb_open_smak.clicked.connect(self.open_smak)
        self.pb_open_imagej.clicked.connect(self.open_imagej)
        self.pb_open_tomviz.clicked.connect(self.open_tomviz)
        self.pb_open_mantis.clicked.connect(self.open_mantis)
        self.pb_open_athena.clicked.connect(self.open_athena)

        self.pb_browse_hdf.clicked.connect(self.browse_file)
        self.pb_load.clicked.connect(self.load_stack)
        self.pb_view_xrf_stack.clicked.connect(self.view_stack)
        self.pb_lauch_db.clicked.connect(self.open_db_tools_3id)
        self.pb_save_up_stack.clicked.connect(self.save_stack)

        self.pb_ref_xanes.clicked.connect(self.select_ref_file)
        self.pb_elist_xanes.clicked.connect(self.select_elist)
        self.pb_plot_xanes_refs.clicked.connect(self.plot_refs)

        # Analysis
        self.pb_pca_scree.clicked.connect(self.pca_scree_)
        self.pb_calc_components.clicked.connect(self.calc_comp_)
        self.pb_kmeans_elbow.clicked.connect(self.kmeans_elbow)
        self.pb_calc_cluster.clicked.connect(self.clustering_)
        self.pb_xanes_fit.clicked.connect(self.fast_xanes_fitting)

        self.show()

        self.le_wd.setText(str(os.getcwd()))

    def onChange(self):
        if self.main_tab.currentIndex() == 0:
            self.pb_xanes_fit.setEnabled(False)
        else:
            self.pb_xanes_fit.setEnabled(True)

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

            logger.Error('Wrong dathena.bat path')

    def open_imagej(self):
        logger.info('opening ImgaeJ')
        try:
            imagej_path = r'C:\Users\pattammattel\Fiji.app\ImageJ-win64.exe'
            Popen([imagej_path])

        except FileNotFoundError:

            logger.Error('Wrong ImageJ.exe path')

    def open_smak(self):
        smak_path = r'C:\Program Files\smak\smak.exe'
        Popen([smak_path], creationflags=CREATE_NEW_CONSOLE)

    def open_mantis(self):

        logger.info('opening  Mantis')
        try:
            mantis_path = r'C:\Users\pattammattel\mantis-2.3.02.amd64.exe'
            Popen([mantis_path])

        except FileNotFoundError:
            logger.Error('Wrong Mantis.exe path')

    def open_tomviz(self):

        logger.info('opening TomViz')

        try:
            tomviz_path = r'C:\Program Files\tomviz\bin\tomviz.exe'
            Popen([tomviz_path, self.le_tiff_file.text()])

        except FileNotFoundError:
            logger.Error('Wrong tomviz.exe path')

    # XRF Loading

    def browse_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Select image data", '', 'image file(*.hdf *.h5 *tiff *tif )')
        self.le_datafile.setText(str(file_name[0]))
        self.load_stack()

    def load_stack(self):
        logger.info('Loading.. please wait...')
        self.xmin, self.xmax = int(self.sb_xrf_x_start.value()), int(self.sb_xrf_x_end.value())
        file_name  = self.le_datafile.text()

        if file_name.endswith('.h5'):
            stack_, mono_e = get_xrf_data(self.xmin, self.xmax, file_name)
            self.sb_xrf_x_end.setValue(mono_e/10)

        elif file_name.endswith('.tiff' or '.tif'):
            stack_ = tf.imread(self.le_datafile.text()).transpose(1, 2, 0)

        logger.info(f' loaded stack with {np.shape(stack_)} from the file')
        self.im_stack = stack_.T
        self.sb_xrf_x_start.setValue(0)
        self.sb_xrf_x_end.setValue(self.im_stack.shape[0])
        logger.info(f' Transposed to shape: {np.shape(self.im_stack)}')

        try:
            self.view_stack()
            self.update_stack_info()

        except NameError:
            logger.error('No image file loaded')
            pass

        logger.info(f'completed image shape {np.shape(self.im_stack)}')


    def update_stack_info(self):
        z, x, y = np.shape(self.updated_stack)
        self.sb_xrange2.setValue(x)
        self.sb_yrange2.setValue(y)
        logger.info('Stack info has been updated')

    def update_stack(self):

        self.updated_stack = remove_nan_inf(self.im_stack)

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
                                           norm_point=int(self.le_norm_point.text()))


        logger.info(f'Updated image is in use')

    def view_stack(self):
        if not self.im_stack.ndim == 3:
            raise ValueError("stack should be an ndarray with ndim == 3")
        else:
            self.update_stack()
            self._new_window = StackSpecViewer(self.updated_stack)
            self._new_window.show()
            logger.info('opening stack viewer')

    def save_stack(self):
        self.update_stack()
        file_name = QFileDialog().getSaveFileName(self, "", '', 'image file(*tiff *tif )')
        tf.imsave(str(file_name[0]), self.updated_stack.transpose(0,2,1))
        logger.info(f'Updated Image Saved: {str(file_name[0])}')

    def open_xrf_stack_imagej(self):
        tf.imsave(f'{self.le_wd.text()}/tmp_image.tiff', np.float32(self.updated_stack), imagej=True)
        imagej_path = r'C:\Users\pattammattel\Fiji.app\ImageJ-win64.exe'
        Popen([imagej_path, f'{self.le_wd.text()}/tmp_image.tiff'])

    # XANES files

    def select_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'text file (*.txt *.csv *.xlsx )')
        self.le_ref.setText(str(file_name[0]))

    def select_elist(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'text file (*.txt *.csv *.xlsx )')
        self.le_elist.setText(str(file_name[0]))
        global energy_
        energy_ = np.loadtxt(self.le_elist.text())

    def plot_refs(self):
        plot_xanes_refs(f=self.le_ref.text())

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

        ims, comp_spec, decon_spec = decompose_stack(self.updated_stack,
            decompose_method=method_ , n_components_=n_components)

        if self.cb_autosave.isChecked():
            dest = str(self.le_wd.text())
            tf.imsave(dest+'/'+ method_+'_components.tiff', np.float32(ims))
            np.savetxt(dest+'/'+ method_+ '_eigen_spectra.txt', comp_spec)
            np.savetxt(dest+'/'+ method_+'_deconv_spectra.txt', decon_spec)

        self._new_window3 = ComponentViewer(ims, comp_spec, decon_spec)
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

        if self.cb_autosave.isChecked():

            dest = str(self.le_wd.text())
            tf.imsave(dest+'/'+ method_+'_clusters.tiff', np.float32(decon_images))
            tf.imsave(dest+'/'+ method_+ '_cluster_map.tiff', np.float32(X_cluster))
            np.savetxt(dest+'/'+ method_+'_deconv_spectra.txt', decon_spectra)

        self._new_window4 = ClusterViewer(decon_images,X_cluster, decon_spectra)
        self._new_window4.show()

        logger.info('Process complete')

    def fast_xanes_fitting(self):

        logger.info('Process started..')

        e_list1 = np.loadtxt(self.le_elist.text()) + int(self.sb_xanes_e_shift.value())
        ref1 = np.loadtxt(self.le_ref.text())
        self.update_stack()

        xanes_maps, ff = xanes_fitting(self.updated_stack, e_list1,ref1, method=self.cb_xanes_fitting_method.currentText())
        logger.info('Process complete')

        if self.cb_autosave.isChecked():
            dest = str(self.le_wd.text())
            tf.imsave(dest+'/XANES_Map.tiff', np.float32(xanes_maps))

        self._new_window5 = ComponentViewer(xanes_maps.T, ff, ff)
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
