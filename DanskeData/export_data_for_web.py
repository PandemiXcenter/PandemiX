# Python copy of export_data_for_web.ipynb, as of 15/04
# %%
from IPython import get_ipython

# %%
# Notebook for exporting particular data for JSON to use in interactive figures
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'widget')
plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams["image.cmap"] = "Dark2"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)
plt.rcParams['lines.markersize'] = 10
# get_ipython().run_line_magic('matplotlib', 'widget')
# plt.style.use('ggplot')
import matplotlib.colors as colors
# cmap = plt.cm.get_cmap('Dark2',len(ageGroups))
from matplotlib import cm # Colormaps

import locale
import matplotlib.dates as mdates
locale.setlocale(locale.LC_TIME,"Danish")
# ax = plt.gca()
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y \n %B'))

import os
# import csv
import math


from datetime import date


saveFigures = True
print('saveFigures is set to: '+str(saveFigures))

print('Done loading packages')

def rnMean(data,meanWidth):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTime(t,meanWidth):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]


# %%
print('Make sure to run "get_data" first, so the most recent data is used')


# %%
ssidatapath = "ssi_data"
rootdir = os.getcwd() +"/" + ssidatapath


for subdir, dirs, files in os.walk(rootdir):
    if not len(files) == 0:
        latestdir = subdir
        latestDate = pd.to_datetime(subdir[-10:])

print(latestdir)
print(latestDate)


# %%
# Load the data
fulldfPCR = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
fulldf = pd.read_csv(latestdir+'/Test_pos_over_time_antigen.csv',delimiter = ';',dtype=str)

# Cut out the last two rows of summary numbers
dfPCR = fulldfPCR.iloc[:-2,:].copy()
df = fulldf.iloc[:-2,:].copy()


# %%

dfPCR["PosPct"] = pd.to_numeric(dfPCR["PosPct"].astype(str).apply(lambda x: x.replace(',','.')))
dfPCR["Date"] = dfPCR["Date"].astype('datetime64[D]')
df["PosPct"] = pd.to_numeric(df["PosPct"].astype(str).apply(lambda x: x.replace(',','.')))
df["Date"] = df["Date"].astype('datetime64[D]')

rows_to_fix_period_in = ["NewPositive","NotPrevPos","PrevPos","Tested","Tested_kumulativ"]
for name in rows_to_fix_period_in:
    df[name] = pd.to_numeric(df[name].astype(str).apply(lambda x: x.replace('.','')))
    dfPCR[name] = pd.to_numeric(dfPCR[name].astype(str).apply(lambda x: x.replace('.','')))


# %%
# col1 = df.Tested
# col2 = df.NewPositive
# colId = df.Date
# # newdf = pd.DataFrame(data = {'Test':col1,'Pos':col2})
# newdf = pd.DataFrame(data = {'colId':colId,'Test':col1,'Pos':col2})
# newdf = newdf.set_index('colId')

# newdf.to_csv('web/AntigenTestsCleaned.csv')
# newdf

df = df.set_index('Date')
dfPCR = dfPCR.set_index('Date')

firstDate = np.datetime64('2021-02-10')
df = df.loc[df.index >= firstDate]
dfPCR = dfPCR.loc[dfPCR.index >= firstDate]


# df.to_csv('web/AntigenTestsCleaned.csv')


# %%
newdf = pd.DataFrame()

newdf['Dato'] = df.index
newdf = newdf.set_index('Dato')

newdf['Antal test, PCR'] = dfPCR.Tested
newdf['Antal test, Antigen'] = df.Tested


newdf['Positiv procent, PCR'] = 100*np.divide(dfPCR.NewPositive,dfPCR.Tested)
newdf['Positiv procent, Antigen'] = 100*np.divide(df.NewPositive,df.Tested)


# %%
newdf.to_csv('web/AntigenTestsCleaned.csv')
newdf.to_csv('web/NumTests.csv',columns=['Antal test, PCR','Antal test, Antigen'])
newdf.to_csv('web/PosPct.csv',columns=['Positiv procent, PCR','Positiv procent, Antigen'])

# newdf.plot()
# plt.plot(dfPCR.index,dfPCR.PosPct)
# plt.plot(df.index,df.PosPct)


# %%
# Load the data
df2 = pd.read_csv(latestdir+'/Antigentests_pr_dag.csv',delimiter = ';')

# # Remove the first two rows which are from april 2020
# df2 = df2.iloc[2:,:].copy()

df2["Dato"] = df2["Dato"].astype('datetime64[D]')

df2 = df2.set_index('Dato')
df2 = df2.loc[df2.index >= firstDate]


# %%



allConf = df2.AGpos_PCRpos + df2.AGposPCRneg
# plt.figure()
# plt.plot(df2.index,100 * np.divide(df2.AGposPCRneg,allConf),label='AG positiv, PCR negativ')

newdf['AG positiv, PCR negativ, ud af alle konfirmede'] = 100 * np.divide(df2.AGposPCRneg,allConf)

newdf.to_csv('web/PCRconfirmed.csv',columns=['AG positiv, PCR negativ, ud af alle konfirmede'])


# %%



