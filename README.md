CBIS_DDSM-File-Sorter

The CBIS_DDSM-File-Sorter is a Python script that allows you to easily find and filter out cropped images from DICOM files. The resulting filtered DICOM files are then organized in a way that is compatible with the code provided below, making it easy to use for your own purposes.
Getting Started

To use the CBIS_DDSM-File-Sorter, follow these steps:

    1.)Clone the repository or download the files onto your local machine.
    2.)Install the necessary Python packages by running the following command: pip install -r requirements.txt
    3.)Open cbis_ddsm_file_sorter.py in your favorite Python editor or IDE.
    4.)Modify the INPUT_PATH variable at the top of the script to the path where your DICOM files are stored.
    5.)Modify the OUTPUT_PATH variable at the top of the script to the path where you want to store the filtered DICOM files.
    6.)Run the script by executing python cbis_ddsm_file_sorter.py in your terminal or Python environment.

The script will search for all DICOM files in the INPUT_PATH directory and its subdirectories. It will then filter out any cropped images and create a new zip file containing the filtered DICOM files in the OUTPUT_PATH directory.

Dataset Information

The CBIS-DDSM dataset is a collection of mammography images and their corresponding masks. The dataset is split into training and testing sets, each of which contains DICOM files stored in their respective folders. The filtered DICOM files created by the CBIS_DDSM-File-Sorter are organized in the same way as the original dataset, making it easy to use in conjunction with the code provided above.

The resulting zip file created by the CBIS_DDSM-File-Sorter is approximately 148MB in size and contains roughly 4000 DICOM files of test and training data. The extracted data is approximately 60GB and is organized in their respective folders.
