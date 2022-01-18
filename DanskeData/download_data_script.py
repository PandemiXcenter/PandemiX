# %%
# Notebook for downloading all data
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


# %%
# Flag for whether all data should be updated, or only recent
downloadAllData = False
downloadAllData = True

# %%
# Make folders if they aren't already there
os.system("mkdir ssi_dashboard_zipped")
os.system("mkdir ssi_dashboard")
os.system("mkdir ssi_vacc_zipped")
os.system("mkdir ssi_vacc")
os.system("mkdir ssi_data_zipped")
os.system("mkdir ssi_data")

# %%
# Define paths
ssidatapath = "ssi_data"
currootdir = os.getcwd() +"/" + ssidatapath

# %%
prevDownloads = os.listdir(currootdir)
mostRecent = prevDownloads[-1]
print(mostRecent)
# curDate = np.datetime64(mostRecent[-2:] + '-' + mostRecent[-5:-3] + '-' + mostRecent[-10:-6])
mostRecentDate = np.datetime64(mostRecent[-10:])
mostRecentDate 
print(mostRecentDate)

# # Overvågningsdata

# %%
get_data = True
ssidatapath = "ssi_data"
rootdir = os.getcwd() +"/" + ssidatapath

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
links = soup.find_all("a", string=lambda text: "data" in str(text).lower())

check_str = "<a href=\"https://files.ssi"
for link in links[3:]: 
    # print('---')
    #print(link)
    if str(link)[:len(check_str)]!=check_str:
        # print("not a file; continues...")
        continue
    # print(link)
    file = link["href"]
    yearPos = file.find('2021')
    
    if yearPos == -1:
        print("2021 not found in link; continues...")
        continue

    curDate = file[yearPos:yearPos+4] + '-' + file[yearPos-2:yearPos] + '-' + file[yearPos-4:yearPos-2] 
    
    # print(file)
    
    # Only download new data
    curDatetime = np.datetime64(curDate)
    if (curDatetime > mostRecentDate):
        print(curDatetime)

        filename = "SSI_data_" + curDate
        zipped_save_path = ssidatapath + "_zipped/" + filename + ".zip"
        extracted_save_path = ssidatapath + "/" + filename

        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)

# %% [markdown]
# # Vaccinedata

# %%
get_data = True
ssivaccpath = "ssi_vacc"
rootdir = os.getcwd() +"/" + ssivaccpath

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

# def get_all_data():
url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-vaccinationsdata"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
links = soup.find_all("a", string=lambda text: "data" in str(text).lower())
# print(links)
check_str = "<a href=\"https://files.ssi"
for link in links[3:]: 
    # print('---')
    #print(link)
    if str(link)[:len(check_str)]!=check_str:
        # print("not a file; continues...")
        continue
    # print(link)
    file = link["href"]
    yearPos = file.find('2021')
    
    if yearPos == -1:
        print("2021 not found in link; continues...")
        continue
    # print(yearPos)
    # print(file[yearPos-4:yearPos+4])
    curDate = file[yearPos:yearPos+4] + '-' + file[yearPos-2:yearPos] + '-' + file[yearPos-4:yearPos-2] 


    # Only download new data
    curDatetime = np.datetime64(curDate)
    if (curDatetime > mostRecentDate):
        print(curDatetime)


        filename = "SSI_vacc_" + curDate
        zipped_save_path = ssivaccpath + "_zipped/" + filename + ".zip"
        extracted_save_path = ssivaccpath + "/" + filename
        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)

# %% [markdown]
# # Dashboard data

# %%
get_data = True
ssidashpath = "ssi_dashboard"
rootdir = os.getcwd() +"/" + ssidashpath

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
# links = soup.find_all("a", string=lambda text: "data" in str(text).lower())
links = soup.find_all("a", string=lambda text: "dash" in str(text).lower())
check_str = "<a href=\"https://files.ssi"
for link in links[1:]: 
    

    if str(link)[:len(check_str)]!=check_str:
        # print("not a file; continues...")
        continue
    file = link["href"]
    yearPos = file.find('2021')
    
    if yearPos == -1:
        print("2021 not found in link; continues...")
        continue
    # print(yearPos)
    # print(file[yearPos-4:yearPos+4])
    curDate = file[yearPos:yearPos+4] + '-' + file[yearPos-2:yearPos] + '-' + file[yearPos-4:yearPos-2] 

    # Only download new data
    curDatetime = np.datetime64(curDate)
    if (curDatetime > mostRecentDate):
        print(curDatetime)

        filename = "SSI_dashboard_" + curDate
        zipped_save_path = ssidashpath + "_zipped/" + filename + ".zip"
        extracted_save_path = ssidashpath + "/" + filename
        
        try:
            download_url(file, zipped_save_path)
            with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
                zipObj.extractall(extracted_save_path)
        except: 
            print(file)
