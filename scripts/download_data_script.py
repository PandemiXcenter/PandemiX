# %%
# Script for downloading all data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import requests
from bs4 import BeautifulSoup 
#import urllib.request
import zipfile
import io
import os
import datetime as dt
# import pycountry as pc
import math

# ----------------------- FLAG FOR DOWNLOADING ALL DATA -----------------------
download_all_data = False
# download_all_data = True
print(f'Download all data: {download_all_data}')
# -----------------------------------------------------------------------------

# %%
# Define paths
rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 

path_data = rootdir_data + "ssi_data"
path_dash = rootdir_data + "ssi_dashboard"
path_vacc = rootdir_data + "ssi_vacc"

currootdir = path_data

# %%
prevDownloads = os.listdir(currootdir)
mostRecent = prevDownloads[-1]
# print(mostRecent)
mostRecentDate = np.datetime64(mostRecent[-10:])
mostRecentDate 
print(f'Most recent data file: {mostRecent}, from date {mostRecentDate}')

# %%
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

# %%
### Overvågningsdata
url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
links = soup.find_all("a", string=lambda text: "data" in str(text).lower())

# check_str = "<a href=\"https://files.ssi"
check_str_over = "<a href=\"https://files.ssi.dk/covid19/overvagning/data/overvaagningsdata"
for link in links[3:]: 
    # print('---')
    # print(link)
    if str(link)[:len(check_str_over)]!=check_str_over:
        # print("not a file; continues...")
        continue
    # print(link)
    file = link["href"]
    # print(file)

    # For all files after 2021-04-19, "-covid19-" is included in link
    # To download earlier files, a different script should be made, in which something else is checked for in link...
    covid19Pos = file.find('-covid19-')
    curDay = file[covid19Pos+9:covid19Pos+11]
    curMonth = file[covid19Pos+11:covid19Pos+13]
    curYear = file[covid19Pos+13:covid19Pos+17]
    curDate = f'{curYear}-{curMonth}-{curDay}'
    curDateTime = np.datetime64(curDate)
    # print(curDateTime)
    
    if download_all_data:

        filename = "\\SSI_data_" + curDate
        zipped_save_path = path_data + "_zipped" + filename + ".zip"
        extracted_save_path = path_data  + filename
        print(f'Downloading: {extracted_save_path}')
        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)
    else:
        if (curDateTime > mostRecentDate):
            
            filename = "\\SSI_data_" + curDate
            zipped_save_path = path_data + "_zipped" + filename + ".zip"
            extracted_save_path = path_data  + filename
            print(f'Downloading: {extracted_save_path}')
            
            try:
                download_url(file, zipped_save_path)
                with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                    zipObj.extractall(extracted_save_path)
            except: 
                print('Error downloading... Filename:')
                print(file)

print('Done downloading "overvågningsdata" files')

# %%
### Dashboard data
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
# links = soup.find_all("a", string=lambda text: "data" in str(text).lower())
links = soup.find_all("a", string=lambda text: "dash" in str(text).lower())
# check_str = "<a href=\"https://files.ssi"
check_str_dash = "<a href=\"https://files.ssi.dk/covid19/overvagning/dashboard/overvaagningsdata-dashboard-"
for link in links[3:]: 
    # print('---')
    # print(link)
    if str(link)[:len(check_str_dash)]!=check_str_dash:
        # print("not a file; continues...")
        continue
    # print(link)
    file = link["href"]
    # print(file)

    # For all files after 2021-04-19, "-covid19-" is included in link
    # To download earlier files, a different script should be made, in which something else is checked for in link...
    covid19Pos = file.find('-covid19-')
    curDay = file[covid19Pos+9:covid19Pos+11]
    curMonth = file[covid19Pos+11:covid19Pos+13]
    curYear = file[covid19Pos+13:covid19Pos+17]
    curDate = f'{curYear}-{curMonth}-{curDay}'
    curDateTime = np.datetime64(curDate)
    # print(curDateTime)
    
    if download_all_data:

        filename = "\\SSI_dashboard_" + curDate
        zipped_save_path = path_dash + "_zipped" + filename + ".zip"
        extracted_save_path = path_dash + filename
        print(f'Downloading: {extracted_save_path}')
        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)
    else:
        if (curDateTime > mostRecentDate):
            
            filename = "\\SSI_dashboard_" + curDate
            zipped_save_path = path_dash + "_zipped" + filename + ".zip"
            extracted_save_path = path_dash + filename
            print(f'Downloading: {extracted_save_path}')
            
            try:
                download_url(file, zipped_save_path)
                with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                    zipObj.extractall(extracted_save_path)
            except: 
                print('Error downloading... Filename:')
                print(file)

print('Done downloading "dashboard" files')

# %%
### Vaccinations 
url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-vaccinationsdata"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
links = soup.find_all("a", string=lambda text: "data" in str(text).lower())
check_str_vacc = "<a href=\"https://files.ssi.dk/covid19/vaccinationsdata/zipfil/vaccinationsdata-dashboard"
for link in links[3:]: 
    # print('---')
    # print(link)
    if str(link)[:len(check_str_vacc)]!=check_str_vacc:
        # print("not a file; continues...")
        continue
    # print(link)
    file = link["href"]
    # print(file)

    # For all files after 2021-04-19, "-covid19-" is included in link
    # To download earlier files, a different script should be made, in which something else is checked for in link...
    covid19Pos = file.find('-covid19-')
    curDay = file[covid19Pos+9:covid19Pos+11]
    curMonth = file[covid19Pos+11:covid19Pos+13]
    curYear = file[covid19Pos+13:covid19Pos+17]
    curDate = f'{curYear}-{curMonth}-{curDay}'
    curDateTime = np.datetime64(curDate)
    # print(curDateTime)
    
    if download_all_data:

        filename = "\\SSI_vacc_" + curDate
        zipped_save_path = path_dash + "_zipped" + filename + ".zip"
        extracted_save_path = path_dash + filename
        print(f'Downloading: {extracted_save_path}')
        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)
    else:
        if (curDateTime > mostRecentDate):
            
            filename = "\\SSI_vacc_" + curDate
            zipped_save_path = path_dash + "_zipped" + filename + ".zip"
            extracted_save_path = path_dash + filename
            print(f'Downloading: {extracted_save_path}')
            
            try:
                download_url(file, zipped_save_path)
                with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                    zipObj.extractall(extracted_save_path)
            except: 
                print('Error downloading... Filename:')
                print(file)

print('Done downloading "vaccination" files')


