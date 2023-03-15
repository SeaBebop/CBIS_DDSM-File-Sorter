# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:15:19 2023

@author: Mario
"""

import os
import pandas as pd

#Objective of this program is to be able to find all the dcm files on any pc

#The main changable thing is the list_dir_name and the data read.csv locations

#Once that is set up, running it will automatically set up all the dcm locations
#With its file size

list_dir_name = "D:\\chi\DDSM\\Mass_train_roi"
data = pd.read_csv(".\\SQL_Training_Test_Metadata\\file_path_with_label_NoI.csv")

size_list_dir = []
finalized_list_dir = [] 

#Will be used for comparsion to find the ROI dcm files
Label_list = []
#Number of Images
Number_of_Images=[]
#The counter is used to align the labels and the File Location together
counter = 0;

print()
#Checker is to test the individual file location and how does it look as a str
checker = ''
for file in data['File_Location']:
    #If statement is checking if there are two files in the folder. If there is,
    #append the name and location of the two files onto the list, 
    #if not then there is only 1 file to append
    
    if(len(os.listdir(list_dir_name + file)) == 2):
        finalized_list_dir.append(list_dir_name + file +'\\'+  
                                  fr'{os.listdir(list_dir_name + file)[0]}')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        finalized_list_dir.append(list_dir_name + file +'\\'+  
                                  fr'{os.listdir(list_dir_name + file)[1]}')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        
        counter+=1
        #checker = list_dir_name + file + r'\\'+ os.listdir(list_dir_name + file)[1]
    else:
        finalized_list_dir.append(list_dir_name + file +'\\'+
                                  fr'{os.listdir(list_dir_name + file)[0]}')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        counter+=1
#print(finalized_list_dir)

"""
file_locations = {'File_Location' : finalized_list_dir}

df = pd.DataFrame(file_locations)

df.to_csv('DCM_File_Location.csv')
"""
#Getting the size in bytes then converting them
#kilobyte = 0.001
for dcm in finalized_list_dir:
    size_list_dir.append(os.path.getsize(dcm))

#Creating a csv of my results
series = {'DCM_File_Path':finalized_list_dir,'DCM_File_Size':size_list_dir,'Label':Label_list,'Number of Images': Number_of_Images}

#df = pd.DataFrame(series)
#df = df.replace('\\\\','\\\\\\\\', regex=True)
#df.index.name = 'id'
#df.to_csv('DCM_File_Locations.csv')




####Testing if it load the images from the list
#
# Test with CBIS_DDSM image
#
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
##from tensorflow.keras.models import Model, Sequential
#import keras
#from keras.models import Model, Sequential
#from keras.layers import Input, Dense, Conv2D
#from keras.layers import MaxPooling2D, UpSampling2D, Flatten, Reshape
#from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import pydicom        # install the pydicom package
from PIL import Image # install the pillow package and it is called PIL.

h = 256
w = 256
ch = 1

print()
#dicomdata = pydicom.read_file('1-2.dcm') # just a mask
"""
This is the example code that tests out dcms
Using the finalized_list_dir I can use my locations
This also means of course I can loop them, show the labels of each of them,etc

"""
print(finalized_list_dir[3])
dicomdata = pydicom.read_file(finalized_list_dir[3])  # masked image
tmp = np.zeros((dicomdata.Rows, dicomdata.Columns), dtype="float32")
tmp = dicomdata.pixel_array/65535.0
    
img = Image.fromarray(tmp)
img_resize = img.resize((h,w), Image.LANCZOS)
tmp2 = img_to_array(img_resize)
data = tmp2.reshape((h,w,ch))
print(np.reshape(data, (h, w)).shape)
plt.imshow(np.reshape(data, (h, w)), cmap='gray')

plt.show()
