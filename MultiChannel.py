import sys, os, json, collections
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic
import pyqtgraph.exporters
import tifffile as tf
from itertools import combinations
import time

ui_path = os.path.dirname(os.path.abspath(__file__))
pg.setConfigOption('imageAxisOrder', 'row-major')

cmap_names = ['CET-L13', 'CET-L14', 'CET-L15']
cmap_combo = combinations(cmap_names, 2)
cmap_label1 = ['red', 'green', 'blue']
cmap_label2 = ['yellow', 'magenta', 'cyan']
cmap_dict = {}
for i, name in zip(cmap_names, cmap_label1):
    cmap_dict[name] = pg.colormap.get(i).getLookupTable(alpha=True)

for i, name in zip(cmap_combo, cmap_label2):
    cmap_dict[name] = (pg.colormap.get(i[0]).getLookupTable(alpha=True) +
                       pg.colormap.get(i[1]).getLookupTable(alpha=True)) // 2


class jsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(jsonEncoder, self).default(obj)


class MultiChannelWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MultiChannelWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'uis/mutlichannel.ui'), self)

        self.canvas = self.img_view.addPlot(title="")
        self.canvas.getViewBox().invertY(True)
        self.canvas.setAspectLocked(True)

        self.cb_choose_color.addItems([i for i in cmap_dict.keys()])

        # connections
        self.actionLoad.triggered.connect(self.createMuliColorAndList)
        self.actionLoad_Stack.triggered.connect(self.createMuliColorAndList)
        self.cb_choose_color.currentTextChanged.connect(self.updateImageDictionary)
        self.pb_update_low_high.clicked.connect(self.updateImageDictionary)
        self.listWidget.itemClicked.connect(self.editImageProperties)
        self.actionLoad_State_File.triggered.connect(self.importState)
        self.actionSave_State.triggered.connect(self.exportState)
        self.actionSave_View.triggered.connect(self.saveImage)

    def generateImageDictionary(self):
        """Creates a dictionary contains image path, color scheme chosen, throshold limits etc.
        when user edits the parameters dictionary will be updated and unwrapped for display later.
        This dictionary is saved as json file while saving the state. Two image loading options are possible.
        User can either select multiple 2D array images or one 3D array (stack)"""

        clickedAction = self.sender()

        if clickedAction.text() == "Load Images":
            # multiple images are selected
            self.loadMultipleImageFiles()

        elif clickedAction.text() == "Load Stack":
            # an image stack is selected
            self.loadAsStack()

    def loadMultipleImageFiles(self):

        filter = "TIFF (*.tiff);;TIF (*.tif)"
        QtWidgets.QFileDialog().setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        # choose mutliple tiff files
        names = QtWidgets.QFileDialog().getOpenFileNames(self, "Open files", " ", filter)
        if names[0]:
            self.image_dict = {}
            # select the file directory. Image files are expected to be in the same folder
            self.imageDir = os.path.dirname(names[0][0])

            # create the dictionary
            for colorName, image in zip(cmap_dict.keys(), names[0]):
                # squeeze to allow with pseudo 3D axis from some tomo recon (eg. 1, 100,100 array)
                im_array = np.squeeze(tf.imread(image))
                # set values for thresholding as image min and max
                low, high = np.min(im_array), np.max(im_array)
                # name of the tiff file is chosen as key for the dictionary,
                # inner keys are properties set for that image
                im_name = os.path.basename(image)
                # construct the dictionary
                self.image_dict[f'{os.path.basename(image)}'] = {'ImageName': im_name,
                                                                 'ImageDir': self.imageDir,
                                                                 'Image':im_array,
                                                                 'Color': colorName,
                                                                 'CmapLimits': (low, high),
                                                                 'Opacity':1.0
                                                                 }
        else:
            pass

    def loadAsStack(self):
        """ construct the dictionary with image +number as the key.
        All other steps are similar to the loadMultipleImageFiles function"""

        filter = "TIFF (*.tiff);;TIF (*.tif)"
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open a Stack", '',
                                                            'TIFF(*tiff)', filter)
        if file_name[0]:
            self.imageDir = os.path.dirname(file_name[0])
            self.image_dict = {}
            im_stack  = np.squeeze(tf.imread(file_name[0]))
            #asset the file is a stack
            assert im_stack.ndim == 3, "Not a stack"
            #construct the dictionary
            for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), im_stack)):
                low, high = np.min(image), np.max(image)
                self.image_dict[f'Image {n+1}'] = {'ImageName': f'Image {n+1}',
                                                'ImageDir': self.imageDir,
                                                'Image':image,
                                                 'Color': colorName,
                                                 'CmapLimits': (low, high),
                                                 'Opacity':1.0
                                                 }


    def loadAnImage(self, image, colormap, cmap_limits, opacity = 1):
        """ load single image and colorbar to the widget. This function will be looped for
        multiple images later
        """
        # get pg image item
        img = pg.ImageItem()
        # add image to the graphicsview widget
        self.canvas.addItem(img)
        # set the color map
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(colormap)), color=colormap)
        #image = np.squeeze(tf.imread(image_path))
        # set image to the image item with cmap
        img.setImage(np.array(image), lut=cmap.getLookupTable(), opacity = opacity)

        # set colorbar for thresholding
        bar = pg.ColorBarItem(
            values=cmap_limits,
            cmap=cmap,
            limits=(0, None),
            orientation='vertical'
        )
        bar.setImageItem(img)
        # set composition mode to plus for overlaying
        img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

    def createMultiColorView(self, image_dictionary):
        """ Function creates multi color image view by taking image
        data and parameters from the dictionary"""

        #clear the plots and list in case of re-loading
        self.canvas.clear()
        self.listWidget.clear()

        # display individual images in for loop
        for path_and_color in image_dictionary.values():
            self.loadAnImage(path_and_color['Image'],
                             cmap_dict[path_and_color['Color']],
                             path_and_color['CmapLimits'],
                             path_and_color['Opacity'])

    def displayImageNames(self, image_dictionary):
        """ Populate the list widget table with image name and color used to plot,
        using image dictionary input"""

        for im_name, vals in image_dictionary.items():
            self.listWidget.addItem(f"{im_name},{vals['Color']}")
            self.listWidget.setCurrentRow(0)


    def createMuliColorAndList(self):
        """ Finally Load Images and poplulate the list widget from the dictionary"""
        with pg.BusyCursor(): # gives the circle showing gui is doing something
            self.generateImageDictionary()
            if self.image_dict:
                self.createMultiColorView(self.image_dict)
                self.displayImageNames(self.image_dict)

            else:
                pass


    def sliderSetUp(self, im_array):
        """ Setting the slider min and max from image values"""

        low = (np.min(im_array) / np.max(im_array)) * 100
        self.sldr_low.setMaximum(100)
        self.sldr_low.setMinimum(low)
        self.sldr_high.setMaximum(100)
        self.sldr_high.setMinimum(low)

    def editImageProperties(self, item):
        """ function to control the assigned properties such as color,
        threshold limits, opacity etc of a single image selected using the list widget item """

        editItem = item.text()
        # get the dictionary key from item text
        editItemName = editItem.split(',')[0]
        editItemColor = editItem.split(',')[1]
        im_array = self.image_dict[editItemName]['Image']
        self.sliderSetUp(im_array)
        setValLow = (self.image_dict[editItemName]['CmapLimits'][0] * 100) / np.max(im_array)
        setValHigh = (self.image_dict[editItemName]['CmapLimits'][1] * 100) / np.max(im_array)
        setOpacity = self.image_dict[editItemName]['Opacity'] * 100
        self.sldr_low.setValue(int(setValLow))
        self.sldr_high.setValue(int(setValHigh))
        self.sldr_opacity.setValue(int(setOpacity))
        self.low_high_vals.setText(f'low:{self.sldr_low.value()},'
                                   f'high:{self.sldr_high.value()}')
        self.cb_choose_color.setCurrentText(editItemColor)

    def updateImageDictionary(self):
        newColor = self.cb_choose_color.currentText()
        editItem = self.listWidget.currentItem().text()
        editRow = self.listWidget.currentRow()
        editItemName = editItem.split(',')[0]
        self.imageDir = self.image_dict[editItemName]['ImageDir']
        im_array = self.image_dict[editItemName]['Image']
        self.sliderSetUp(im_array)
        cmap_limits = (self.sldr_low.value() * np.max(im_array) / 100,
                       self.sldr_high.value() * np.max(im_array) / 100)
        self.low_high_vals.setText(f'low:{cmap_limits[0]:.2f},high:{cmap_limits[1]:.2f}')
        opacity = self.sldr_opacity.value()/100
        self.opacity_val.setText(str(opacity))
        self.image_dict[editItemName] = {'ImageName': editItemName,
                                         'ImageDir': self.imageDir,
                                         'Image':im_array,
                                         'Color': newColor,
                                         'CmapLimits': cmap_limits,
                                         'Opacity':opacity
                                         }

        self.createMultiColorView(self.image_dict)
        self.displayImageNames(self.image_dict)
        self.listWidget.setCurrentRow(editRow)

    def exportState(self):

        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save Current State", 'multicolor_params.json',
                                                            'json file(*json)')
        '''
        for val in self.image_dict.values():
            val['CmapLimits'] = json.dumps(str(val['CmapLimits']))
        '''

        if file_name[0]:

            with open(f'{file_name[0]}', 'w') as fp:
                json.dump(self.image_dict, fp, indent=4, cls = jsonEncoder)

        else:
            pass

    def importState(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open a State File", '',
                                                            'json file(*json)')
        if file_name[0]:
            with open(file_name[0], 'r') as fp:
                self.image_dict = json.load(fp)


            self.createMultiColorView(self.image_dict)
            self.displayImageNames(self.image_dict)
        else:
            pass

    def saveImage(self):
        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save Image", 'multicolor_image.png',
                                                            'PNG(*.png);; TIFF(*.tiff);; JPG(*.jpg)')
        exporter = pg.exporters.ImageExporter(self.canvas.getViewBox())
        exporter.export(file_name[0])



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MultiChannelWindow()
    window.show()
    sys.exit(app.exec_())
