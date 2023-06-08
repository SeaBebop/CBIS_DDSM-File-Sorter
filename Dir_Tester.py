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
from zipfile import ZIP_STORED
from tkinter.filedialog import askdirectory
from time import sleep
from tqdm import tqdm
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
#Objective of this program is:

*To be able to find all the dcm files on any pc
*Filter out all the cropped images
*Create a zip file of the filtered dcm into a zip file
*That zip file is organized in a why that it still works in the code down below
*Test with CBIS_DDSM image"

>The zip file is 148mb and the extracted data is roughly 39.5MB~ of DCM organized in their respective folders
>Roughly 700 DCM of test and training data

NOTE: If somehow the dcm files are still incorrect,swap the sign on lines 130-135

"""

print('\nA folder tab just opened! Select the folder that is one place away from the CBIS-DDSM folder')
print('Example: if your CBIS-DDSM location is D:\chi\DDSM\Mass_train_roi\CBIS-DDSM')
print('Then select D:\chi\DDSM\Mass_train_roi')

UserInput= input("Would you like to create a reduced_folder?? The reduced extracted data is 39.5MB~ and the zip file is uncompressed(y/n): ").lower().strip() == 'y'

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
        folder.append(file[1:])
        
        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/') +'/'+  
                                  fr'{os.listdir(list_dir_name + file)[0]}')
        
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        Pathology.append(data['pathology'][counter])        

        finalized_list_dir.append(list_dir_name + file[1:].replace('\\' ,'/')  +'/'+  
                                  fr'{os.listdir(list_dir_name + file)[1]}')
        folder.append(file[1:])
        Label_list.append(data['ROI_mask_file_path'][counter])
        Number_of_Images.append(data['Number_of_Images'][counter])
        Pathology.append(data['pathology'][counter])        
        
        counter+=1
        #checker = list_dir_name + file + r'\\'+ os.listdir(list_dir_name + file)[1]
    else:
        folder.append(file[1:])
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
    
    if(UserInput == True):
    #Folder location is based on where CBIS-DDSM is as previousily selected
    #Folder size should be 39.5 MB with 700 files that are inside 700 folders
    
        print('Creating a folder name reduced file in the \CBIS-DDSM directory. . .')
        
        if not os.path.exists(list_dir_name + '/reduced_files'):
            os.mkdir(list_dir_name + '/reduced_files')
            for dcmIndex in tqdm(range(len(edited_df))):
                os.mkdir( list_dir_name + '/reduced_files' + "/" + edited_df['Label'].iloc[dcmIndex])
                folder_name =  list_dir_name + '/reduced_files' + "/" + edited_df['Label'].iloc[dcmIndex]
                print(folder_name)
                shutil.copy(edited_df['DCM_File_Path'].iloc[dcmIndex], folder_name)
        else:
            print('Error: reduced_file folder already created, if you want to remake it, delete that folder')
                
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
    from tensorflow.keras.regularizers import l1
    import matplotlib.pyplot as plt
    import pandas as pd
    import pydicom        # install the pydicom package
    from PIL import Image # install the pillow package and it is called PIL.
    from sklearn.model_selection import train_test_split

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
    
    #######Working pr
    #######Basic autoencoder example from https://blog.keras.io/building-autoencoders-in-keras.html
    # This is the size of our encoded representations
    dim_1 = 32768/16
    dim_2 = dim_1/64
    dim_3 = dim_1/128
  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    #Change to fit our current expected input (256,256,1)
    n_input = 256*256*1
    # This is our input image
    input_img = keras.Input(shape=(n_input,))
    # "encoded" is the encoded representation of the input
    encoded_1 = layers.Dense(dim_1, activation='relu',activity_regularizer=l1(0.001))(input_img)
    encoded_2 = layers.Dense(dim_2, activation='relu',activity_regularizer=l1(0.001))(encoded_1)
    # "decoded" is the lossy reconstruction of the input
    bottleneck = layers.Dense(dim_3, activation='relu',activity_regularizer=l1(0.001))(encoded_2)
    
    decoded_1 = layers.Dense(dim_2, activation='relu',activity_regularizer=l1(0.001))(bottleneck)
    decoded_2 = layers.Dense(dim_1, activation='relu',activity_regularizer=l1(0.001))(decoded_1)
    output = layers.Dense(n_input, activation='sigmoid')(decoded_2)
    
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, output)
    """
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, bottleneck)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(dim_3,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer_1 = autoencoder.layers[-3]
    decoder_layer_2 = autoencoder.layers[-2](decoder_layer_1)
    decoded =  autoencoder.layers[-1](decoder_layer_2)
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoded)
    """
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
  
    history = autoencoder.fit(x_train, x_train,
                        epochs=200,batch_size=16,
                       
                        shuffle=True,
                        validation_data=(x_test, x_test))
    encoder = keras.Model(input_img, bottleneck)
    
    autoencoder.save('autoencoder.h5')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
   
"""
#Simple Undercomplete
import tensorflow  as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
(x_train,_), (x_test,_)= mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))## Shape of the x_train is (60000, 784)
## Shape of the x_test is (10000,784)
input_l=Input(shape=(784,))
bottleneck=Dense(32, activation='relu')(input_l)
output_l=Dense(784, activation='sigmoid')(bottleneck)
autoencoder=Model(inputs=[input_l],outputs=[output_l])    ## Building the entire autoencoder
encoder=Model(inputs=[input_l],outputs=[bottleneck])    ## Building the encoder
encoded_input=Input(shape=(32,))
decoded=autoencoder.layers[-1](encoded_input)
decoder=Model(inputs=[encoded_input],outputs=[decoded])      ##Building the decoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
"""
#sparse autoencoder
"""
import tensorflow  as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.regularizers import l1
(x_train,_), (x_test,_)= mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
## Shape of the x_train is (60000, 784)
## Shape of the x_test is (10000,784)
input_l=Input(shape=(784,))
encoding_1=Dense(256, activation='relu', activity_regularizer=l1(0.001))(input_l)
bottleneck=Dense(32, activation='relu', activity_regularizer=l1(0.001))(encoding_1)
decoding_1=Dense(256, activation='relu', activity_regularizer=l1(0.001))(bottleneck)
output_l=Dense(784, activation='sigmoid')(decoding_1)
autoencoder=Model(inputs=[input_l],outputs=[output_l])
encoder=Model(inputs=[input_l],outputs=[bottleneck])
encoded_input=Input(shape=(32,))
decoded_layer_2=autoencoder.layers[-2](encoded_input)
decoded=autoencoder.layers[-1](decoded_layer_2)
decoder=Model(inputs=[encoded_input],outputs=[decoded])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,

"""
#Denoising autoencoder, most interested in this one
"""
import tensorflow  as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.regularizers import l1

input_l=Input(shape=(28,28,1))
encoding_1=Conv2D(32, (3,3), activation='relu',padding='same')(input_l)
maxp_1=MaxPooling2D((2,2), padding='same')(encoding_1)
encoding_2=Conv2D(16, (3,3), activation='relu',padding='same')(maxp_1)
maxp_2=MaxPooling2D((2,2), padding='same')(encoding_2)
encoding_3=Conv2D(8, (3,3), activation='relu',padding='same')(maxp_2)
bottleneck=MaxPooling2D((2,2), padding='same')(encoding_3)
decoding_1=Conv2D(8, (3,3), activation='relu', padding='same')(bottleneck)
Up_1=UpSampling2D((2,2))(decoding_1)
decoding_2=Conv2D(16, (3,3), activation='relu', padding='same')(Up_1)
Up_2=UpSampling2D((2,2))(decoding_2)
decoding_3=Conv2D(32, (3,3), activation='relu')(Up_2)
Up_3=UpSampling2D((2,2))(decoding_3)
output_l= Conv2D(1,(3,3),activation='sigmoid',padding='same')(Up_3)
autoencoder=Model(inputs=[input_l],outputs=[output_l])
encoder=Model(inputs=[input_l],outputs=[bottleneck])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
"""
