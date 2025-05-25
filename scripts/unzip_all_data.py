# %%
# Script for unzipping all data
import zipfile
import io
import os
from tqdm import tqdm 

# Define paths
rootdir_data = os.getcwd() +"/../DanskeData/" 

path_data = rootdir_data + "ssi_data"
path_dash = rootdir_data + "ssi_dashboard"
path_vacc = rootdir_data + "ssi_vacc"

# %%
# Overvågningsdata
curRoot = path_data
curFiles = os.listdir(curRoot+'_zipped')

for thisFile in tqdm(curFiles):

    filename = "/" + thisFile[:-4]
    zipped_save_path = curRoot + "_zipped" + filename + ".zip"
    extracted_save_path = curRoot  + filename
    # print(f'Extracting: {extracted_save_path}')
    
    with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
        zipObj.extractall(extracted_save_path)
print('Done extracting all "overvågningsdata" files')

# %%
# Dashboard
curRoot = path_dash
curFiles = os.listdir(curRoot+'_zipped')

for thisFile in tqdm(curFiles):

    filename = "/" + thisFile[:-4]
    zipped_save_path = curRoot + "_zipped" + filename + ".zip"
    extracted_save_path = curRoot  + filename
    # print(f'Extracting: {extracted_save_path}')
    
    with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
        zipObj.extractall(extracted_save_path)
print('Done extracting all "dashboard" files')

# %%
# Vaccinations
curRoot = path_vacc
curFiles = os.listdir(curRoot+'_zipped')

for thisFile in tqdm(curFiles):

    filename = "/" + thisFile[:-4]
    zipped_save_path = curRoot + "_zipped" + filename + ".zip"
    extracted_save_path = curRoot  + filename
    # print(f'Extracting: {extracted_save_path}')
    
    with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
        zipObj.extractall(extracted_save_path)
print('Done extracting all "vaccination" files')


