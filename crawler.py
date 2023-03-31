from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import pandas as pd
import os

# Function for grabbing the creditcard.csv from kaggle if it doesnt exist in the current directory
# and returns a pandas dataframe containing the data from the csv file.
def crawler():
    if not os.path.exists("data/creditcard.csv"): # Dont fetch new file if its already here.
        api = KaggleApi()
        api.authenticate()
        #downloading the credit card dataset
        api.dataset_download_file('deepfake-detection-challenge/data', "deepfake-detection-challenge.zip")
        zf = ZipFile('deepfake-detection-challenge.zip')
        #extracted data is saved in the same directory
        zf.extractall() 
        zf.close()
        os.remove("deepfake-detection-challenge.zip") ## remove zip file after extracting.
    return pd.read_csv("deepfake-detection-challenge.csv") ## return the data from file.