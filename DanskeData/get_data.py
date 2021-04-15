# Python file copy of get_data.ipynb, 15/04-2021
# %%
# A copy of the top of Christians script to load the data

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
import pycountry as pc
import math

#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

import locale
locale.setlocale(locale.LC_TIME,"Danish")


# %%
# Danish regional data:

get_data = True
ssidatapath = "ssi_data"
rootdir = os.getcwd() +"/" + ssidatapath


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def get_all_data():
    url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata"
    #url = "http://www.ssi.dk/covid19/overvagning/data/data-epidemiologisk(e)-rapport-"
    #old link#url = "https://www.ssi.dk/sygdomme-beredskab-og-forskning/sygdomsovervaagning/c/covid19-overvaagning/arkiv-med-overvaagningsdata-for-covid19"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all("a", string=lambda text: "data" in str(text).lower())
    #print(links.index("<a href=\"http://www.ssi.dk/covid19data\" target=\"_blank\">www.ssi.dk/covid19data</a>"))
    #print(links)
    check_str = "<a href=\"https://files.ssi"
    for link in links[3:]: 
        #print(link)
        if str(link)[:len(check_str)]!=check_str:
            print("not a file; continues...")
            continue
        #print(link)
        file = link["href"]
        old_date = str(file).split("-")[-2]
        if len(old_date)!=8:
            print("not a date; continues...")
            continue
        new_date = old_date[4:] + "-" + old_date[2:4] + "-" + old_date[0:2]
        filename = "SSI_data_" + new_date
        zipped_save_path = ssidatapath + "_zipped/" + filename + ".zip"
        extracted_save_path = ssidatapath + "/" + filename
        
        download_url(file, zipped_save_path)
        with zipfile.ZipFile(zipped_save_path, 'r') as zipObj:
            zipObj.extractall(extracted_save_path)

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + pd.DateOffset(days=n)


# %%
# start_dt = pd.to_datetime("2020-03-01")
# end_dt = pd.to_datetime("2020-05-01")

#for date in daterange(start_dt, end_dt):
    #print(date)
#    print(dt.strftime("%Y-%m-%d"))

if not os.path.exists('figs'):
    os.makedirs('figs')

if get_data:
    os.system("mkdir ssi_data_zipped")
    os.system("mkdir ssi_data")
    get_all_data()

dk_cases_by_age_df = pd.DataFrame()
for subdir, dirs, files in os.walk(rootdir):
    if not len(files) == 0:
        date = pd.to_datetime(subdir[-10:])
        for file in files:
            if file.lower() == "cases_by_age.csv":
                cases_age = pd.read_csv(subdir + "/" + file, sep=";", decimal=",")
                cases_age["Dato"] = date
                cases_age["Antal_bekræftede_COVID-19"] = pd.to_numeric(cases_age["Antal_bekræftede_COVID-19"].astype(str).apply(lambda x: x.replace('.','')))
                cases_age["Antal_testede"] = pd.to_numeric(cases_age["Antal_testede"].astype(str).apply(lambda x: x.replace('.','')))
                dk_cases_by_age_df = dk_cases_by_age_df.append(cases_age, ignore_index=True)
dk_cases_by_age_df = dk_cases_by_age_df.sort_values(by=['Dato', "Aldersgruppe"]).reset_index(drop=True)
dk_cases_by_age_df#.columns
testede_alder = dk_cases_by_age_df.groupby(["Dato", "Aldersgruppe"]).sum()["Antal_testede"].unstack()
bekr_alder = dk_cases_by_age_df.groupby(["Dato", "Aldersgruppe"]).sum()["Antal_bekræftede_COVID-19"].unstack()

# uncomment for daily data:
testede_alder = testede_alder.diff()
bekr_alder = bekr_alder.diff()

if get_data:
    print("Data is loaded")


# %%
print('Done loading all data')


