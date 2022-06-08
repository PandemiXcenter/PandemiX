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
import sys, getopt

# ----------------------- FLAG FOR DOWNLOADING ALL DATA -----------------------
download_all_data = False
# download_all_data = True
print(f'Download all data: {download_all_data}')
# -----------------------------------------------------------------------------

argv = sys.argv[1:]
countries_list = "" 
variants_list = "" 
dest_directory = "/../CoV-Spectrum/" 
help_str = 'python3 download_cov-spect_data.py -c <countries(no-space-comma-sep.)> -v <variants(no-space-comma-sep.)> \n\nEx:\npython3 download_cov-spect_data.py -c Denmark,Sweden -v b.1.1.7,BA.1'

try:
    opts, args = getopt.getopt(argv,"hc:v:d:",["countries=", "variants=", "directory="])
except getopt.GetoptError:
    print(help_str)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(help_str)
        sys.exit()
    elif opt in ("-c", "--countries"):
        countries_list = arg.split(",")
    elif opt in ("-v", "--variants"):
        variants_list = arg.split(",")
    elif opt in ("-d", "--directory"):
        dest_directory = arg

print()
print('Countries chosen for downloads: ')
[print(c) for c in countries_list]

print()
print('Variants chosen for downloads: ')
[print(v) for v in variants_list]

# Define paths
rootdir_data = os.getcwd() + dest_directory

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

url = lambda c, v: f"https://cov-spectrum.org/explore/{c}/AllSamples/AllTimes/variants?pangoLineage={v}&#"

for c, v in [(c, v) for c in countries_list for v in variants_list]:
    u = url(c, v)
    download_url(u, rootdir_data + f"numSeq_{c}_{v}")
    print(u) 



