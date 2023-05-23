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

>The zip file is 148mb and the extracted data is roughly 2gb~ of DCM organized in their respective folders
>Roughly 4000 DCM of test and training data

"""

print('\nA folder tab just opened! Select the folder that is one place away from the CBIS-DDSM folder')
print('Example: if your CBIS-DDSM location is D:\chi\DDSM\Mass_train_roi\CBIS-DDSM')
print('Then select D:\chi\DDSM\Mass_train_roi')

zipUserInput= input('Would you like to create a zip file? The reduced extracted data is 2gb~ and the zip file is (y/n): ').lower().strip() == 'y'

path = askdirectory(title='Select Folder') 
#print(path)  

cleaned_path = path.replace(r'/',r"\\" )
#print(cleaned_path)         
   
list_dir_name = path
data = pd.read_csv("./SQL_Training_Test_Metadata/file_path_with_label_NoI_pathology.csv")

size_list_dir = []
finalized_list_dir = [] 

#Will be used for comparsion to find the ROI dcm files
Label_list = []
#Number of Images
Number_of_Images=[]
Pathology=[]
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
        Pathology.append(data['pathology'][counter])        

        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/')  +'/'+  
                                  fr'{os.listdir(list_dir_name + file)[1]}')
        folder.append(list_dir_name + file[1:].replace('\\' ,'/') +'/')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        Pathology.append(data['pathology'][counter])        
        
        counter+=1
        #checker = list_dir_name + file + r'\\'+ os.listdir(list_dir_name + file)[1]
    else:
        folder.append(list_dir_name + file[1:].replace('\\' ,'/') +'/')
        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/') +'/'+  
                             fr'{os.listdir(list_dir_name + file)[0]}')
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        Pathology.append(data['pathology'][counter])
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
                  'Label':Label_list,'Number of Images': Number_of_Images,'Folder':folder,'Classification': Pathology}
    
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
    edited_df = df.loc[df['DCM_File_Size'] < 98000]
    edited_df.to_csv('DCM_File_Paths_Reduced.csv')
    
    #print(edited_df['DCM_File_Path'].iloc[0])
   
    #This takes in the reduced DCM ROI to a zip file
    
    if(zipUserInput == True):
        with ZipFile('dcm.zip','w', compression= ZIP_DEFLATED) as z:
            for dcmIndex in tqdm(range(len(edited_df))):
                z.write(edited_df['DCM_File_Path'].iloc[dcmIndex])
                
    else:
        print('Okay, zip file was not created')
            
    
    #Testing if it load the images from the list
    
    #Test with CBIS_DDSM image
    
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import Model, Sequential
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Conv2D
    from keras.layers import MaxPooling2D, UpSampling2D, Flatten, Reshape
    #from keras.preprocessing.image import load_img, img_to_array
    import matplotlib.pyplot as plt
    import pandas as pd
    import pydicom        # install the pydicom package
    from PIL import Image # install the pillow package and it is called PIL.
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import plot_model
    from matplotlib import pyplot
    # train autoencoder for classification with no compression in the bottleneck layer
    import keras
    from keras import layers
    h = 256
    w = 256
    ch = 1
    



    #This is the example code that tests out dcms
    #Using the finalized_list_dir I can use my locations
    #This also means of course I can loop them, show the labels of each of them,etc
    
    trigger = 0
    counter = 0
    dcmMask = np.zeros((len(edited_df),h,w,1), dtype="uint8")
    y_label = []

    
    for i in tqdm(range(0,len(edited_df))):
        #Setting up x and y
        dicomdata = pydicom.read_file(edited_df['DCM_File_Path'].iloc[i],force=True)  # masked image
        y_label.append(edited_df['Classification'].iloc[i]) 
        #Testing if labels and dcm align properly 
        #if trigger < 5:
        #    print('This is class ',i+20,edited_df['Classification'].iloc[i+20])
        #    print('This is path ', i+20,edited_df['DCM_File_Path'].iloc[i+20])
        #    trigger+=1
        #Converting to numpy array
        tmp = np.zeros((dicomdata.Rows, dicomdata.Columns), dtype="float32")
        tmp = dicomdata.pixel_array/65535.0
            
        img = Image.fromarray(tmp)
        img_resize = img.resize((h,w), Image.LANCZOS)
        tmp2 = img_to_array(img_resize)
        dcmMask[i] = tmp2.reshape((h,w,ch))
        #Testing if data is set up as a 3d Rensor
        #if trigger == 0:
        #    print('this is dimension',dcmMask[i].ndim)
        #    print('this is shape',dcmMask[i].shape)
        #    print('this is type',dcmMask[i].dtype)
        #    print('this looks like this', dcmMask[i])
        #    trigger+=1
        #If you are interested to see all the pictures individually,increased runtime
        #data = tmp2.reshape((h,w,ch))
        #plt.imshow(np.reshape(data, (h, w)), cmap='gray')
        #plt.show()
    #print(dcmMask.shape)

    x_train, x_test, y_train, y_test = train_test_split(dcmMask, y_label, test_size=0.30, random_state=7)

    #Normalize the array to 0 and 1
    #print(x_train.shape)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    #print(x_train.shape)
    
    #######Basic autoencoder example from https://blog.keras.io/building-autoencoders-in-keras.html
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    #Change to fit our current expected input (256,256,1)
    n_input = 256*256*1
    # This is our input image
    input_img = keras.Input(shape=(n_input,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(n_input, activation='sigmoid')(encoded)
    
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    plot_model(autoencoder, 'autoencoder_no_compress.png', show_shapes=True)    
    history = autoencoder.fit(x_train, x_train,
                        epochs=10,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
