# MIDAS (Multimodal Image Data Analysis Software)
A software suite to load and analysis imaging and spectrosocpy data collected at NSLS-II, Brookhaven National Laboratory


Installation

1. install python 3.7 (No tested for later versions yet). Make sure you check 'ADD TO PATH' during the installation.
	If you already have this skip to next step.
2. dowanload the midas program to a folder 
3. open a terminal program in the folder. (if not familiar check it here: https://www.groovypost.com/howto/open-command-window-terminal-window-specific-folder-windows-mac-linux/)
4. type 'pip install -r requirements.txt'
5. after all the downloads without errors, type ' python main.py'. Progam should appear now.

![alt text](https://github.com/pattammattel/NSLS-II-MIDAS/blob/main/Midas_view.JPG)

 
 # Loading an Image
 
 ## Supported Formats
 
 ### XRF Data
 
 MIDAS can unpack fluorescence **.h5 data** from NSLS-II beamlines, HXN, XFM, SRX and TES.XRF data will be normalized with I0 (scalar) value on the file. 
 
 ### 2D-XANES Data
 
 2D and 3D **tiff** images (stacks). List of energies has to be loaded as a seperate tiff file.
 
 
To open above data types use the file menu and select the file. Once loaded correctly you should see the image on the top panel and the spectrum on the bottom panel. In case of incorrect formatting or unsupported formats the bottom left corner of the program show the error. 
