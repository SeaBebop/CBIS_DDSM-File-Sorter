# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:15:19 2023

@author: Mario
"""
import shutil
import os
import pandas as pd
from tkinter import Tk
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
from tkinter.filedialog import askdirectory
from time import sleep
from tqdm import tqdm



"""
#Objective of this program is:

*To be able to find all the dcm files on any pc
*Filter out all the cropped images
*Create a zip file of the filtered dcm into a zip file
*That zip file is organized in a why that it still works in the code down below
*Test with CBIS_DDSM image"

>The zip file is 148mb and the extracted data is roughly 60gb~ of DCM organized in their respective folders
>Roughly 4000 DCM of test and training data

"""

print('\nA folder tab just opened! Select the folder that is one place away from the CBIS-DDSM folder')
print('Example: if your CBIS-DDSM location is D:\chi\DDSM\Mass_train_roi\CBIS-DDSM')
print('Then select D:\chi\DDSM\Mass_train_roi')

zipUserInput= input('Would you like to create a zip file? The reduced extracted data is 60gb~ and the zip file is (y/n): ').lower().strip() == 'y'

path = askdirectory(title='Select Folder') 
#print(path)  

cleaned_path = path.replace(r'/',r"\\" )
#print(cleaned_path)         
   
list_dir_name = path
data = pd.read_csv("./SQL_Training_Test_Metadata/file_path_with_label_NoI.csv")

size_list_dir = []
finalized_list_dir = [] 

#Will be used for comparsion to find the ROI dcm files
Label_list = []
#Number of Images
Number_of_Images=[]
folder = []
#The counter is used to align the labels and the File Location together
counter = 0;

#print()
#Checker is to test the individual file location and how does it look as a str
checker = ''
for file in data['File_Location']:
    #If statement is checking if there are two files in the folder. If there is,
    #append the name and location of the two files onto the list, 
    #if not then there is only 1 file to append
    
    if(len(os.listdir(list_dir_name + file)) == 2):
        folder.append(list_dir_name + file[1:].replace('\\' ,'/') +'/')
        
        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/') +'/'+  
                                  fr'{os.listdir(list_dir_name + file)[0]}')
        
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        
        
        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/')  +'/'+  
                                  fr'{os.listdir(list_dir_name + file)[1]}')
        folder.append(list_dir_name + file[1:].replace('\\' ,'/') +'/')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        
        
        counter+=1
        #checker = list_dir_name + file + r'\\'+ os.listdir(list_dir_name + file)[1]
    else:
        folder.append(list_dir_name + file[1:].replace('\\' ,'/') +'/')
        finalized_list_dir.append(list_dir_name + file +'\\'+
                                  fr'{os.listdir(list_dir_name + file)[0]}')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        counter+=1
else:
    #print(finalized_list_dir)
    
    """
    file_locations = {'File_Location' : finalized_list_dir}
    
    df = pd.DataFrame(file_locations)
    
    df.to_csv('DCM_File_Location.csv')
    """
    #Getting the size in bytes then converting them
    #kilobyte = 0.001
    #print(len(folder))
    #print(len(finalized_list_dir))
    #print(counter)
    for dcm in finalized_list_dir:
        size_list_dir.append(os.path.getsize(dcm))

    #Creating a csv of my results
    dictionary = {'DCM_File_Path':finalized_list_dir,'DCM_File_Size':size_list_dir,
                  'Label':Label_list,'Number of Images': Number_of_Images,'Folder':folder}
    
    #df = pd.DataFrame(series)
    #df = df.replace('\\\\','\\\\\\\\', regex=True)
    #df.index.name = 'id'
    #df.to_csv('DCM_File_Locations.csv')
    df = pd.DataFrame(dictionary)
    #print(cleaned_path)
    #print(path)

   
    df = df.replace('.','')
    #This is filter for only the ROI
    #edited_df = df[df['DCM_File_Size'] > 1130000]
    edited_df = df.loc[df['DCM_File_Size'] > 9865000]
    edited_df.to_csv('DCM_File_Paths_Reduced.csv')
    
    #print(edited_df['DCM_File_Path'].iloc[0])
   
    #This takes in the reduced DCM ROI to a zip file

    """
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

    This is the example code that tests out dcms
    Using the finalized_list_dir I can use my locations
    This also means of course I can loop them, show the labels of each of them,etc
    

    counter = 0
    train_data = np.zeros((len(finalized_list_dir),h,w,1), dtype="uint8")
    
    for i in tqdm(range(0,len(finalized_list_dir))):
      
        dicomdata = pydicom.read_file(finalized_list_dir[i])  # masked image
        tmp = np.zeros((dicomdata.Rows, dicomdata.Columns), dtype="float32")
        tmp = dicomdata.pixel_array/65535.0
            
        img = Image.fromarray(tmp)
        img_resize = img.resize((h,w), Image.LANCZOS)
        tmp2 = img_to_array(img_resize)
        train_data[i] = tmp2.reshape((h,w,ch))
    
        #If you are interested to see all the pictures individually,increased runtime
        #data = tmp2.reshape((h,w,ch))
        #plt.imshow(np.reshape(data, (h, w)), cmap='gray')
        #plt.show()
"""
