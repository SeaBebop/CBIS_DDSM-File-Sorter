CBIS_DDSM-File-Sorter

The CBIS_DDSM-File-Sorter is a Python script that allows you to easily find and filter out cropped images from DICOM files. The resulting filtered DICOM files are then organized in a way that is compatible with the code provided below, making it easy to use for your own purposes.
Getting Started

To use the CBIS_DDSM-File-Sorter, follow these steps:

    1.)Clone the repository or download the files onto your local machine.
    2.)Open Dir_Tester.py in your favorite Python editor 
    or IDE(Anaconda/Google Colab is what was used).
    3.)Modify the INPUT_PATH variable at the top of the script to 
    the path where your DICOM files are stored.
    4.)Modify the OUTPUT_PATH variable at the top of the script to 
    the path where you want to store the filtered DICOM files.
    5.)Run the script by executing python Dir_Tester.py 
    in your terminal or Python environment.
    
Compatibility

The resulting zip file created by the CBIS_DDSM-File-Sorter is compatible with the code provided below, which is used to load and preprocess the CBIS-DDSM dataset.

python

    import numpy as np
    import pydicom
    import os
    # Test with CBIS_DDSM image
    #
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import Model, Sequential
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Conv2D
    from keras.layers import MaxPooling2D, UpSampling2D, Flatten, Reshape
    from keras.preprocessing.image import load_img, img_to_array
    import matplotlib.pyplot as plt
    import pandas as pd
    import pydicom        # install the pydicom package
    from PIL import Image # install the pillow package and it is called PIL.
    
    h = 256
    w = 256
    ch = 1
    
    print()
    dicomdata = pydicom.read_file('1-2.dcm') # just a mask

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
        data = tmp2.reshape((h,w,ch))
        plt.imshow(np.reshape(data, (h, w)), cmap='gray')
        plt.show()

The script will search for all DICOM files in the INPUT_PATH directory and its subdirectories. It will then filter out any cropped images and create a new zip file containing the filtered DICOM files in the OUTPUT_PATH directory.

Dataset Information

The CBIS-DDSM dataset is a collection of mammography images and their corresponding masks. The dataset is split into training and testing sets, each of which contains DICOM files stored in their respective folders. The filtered DICOM files created by the CBIS_DDSM-File-Sorter are organized in the same way as the original dataset, making it easy to use in conjunction with the code provided above.

The resulting zip file created by the CBIS_DDSM-File-Sorter is approximately 148MB in size and contains roughly 4000 DICOM files of test and training data. The extracted data is approximately 60GB and is organized in their respective folders.
