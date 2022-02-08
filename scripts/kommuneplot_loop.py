# %%
# Load packages and settings
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)
import seaborn as sns


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams["image.cmap"] = "tab10"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
fs_label = 16
parameters = {
                'axes.labelsize': fs_label,
                'axes.titlesize': fs_label+4,
                'xtick.labelsize': fs_label,
                'ytick.labelsize': fs_label, 
                'legend.fontsize': fs_label, 
                'lines.markersize': 10,
                'lines.linewidth': 3
             }
plt.rcParams.update(parameters)

from matplotlib import cm # Colormaps
import matplotlib.colors as colors
# cmap = plt.cm.get_cmap('Dark2',len(ageGroups))

import locale
import matplotlib.dates as mdates
locale.setlocale(locale.LC_TIME,"Danish")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
# ax1.spines['top'].set_visible(False) 

import os
# import csv
import math

from datetime import date


saveFigures = True
# saveFigures = False
print('saveFigures is set to: '+str(saveFigures))

print('Done loading packages')

# Define running mean functions
def rnMean(data,meanWidth):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTime(t,meanWidth):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]

# %%
%matplotlib widget

# %%
# Define paths
rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 

path_data = rootdir_data + "ssi_data\\"
path_dash = rootdir_data + "ssi_dashboard\\"
path_vacc = rootdir_data + "ssi_vacc\\"
path_figs = os.getcwd() +"\\..\\Figures\\" 

# %%
# Walk to relavant folder
latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
df = pd.read_csv(latestdir+'/Kommunalt_DB/07_bekraeftede_tilfaelde_pr_dag_pr_kommune.csv',encoding='latin1',delimiter = ';')

# %%
# df
# kommune_df
df.Kommunenavn.unique()
# (kommune_df['Bekræftede tilfælde']/antal_borgere(curKommune))*100

# %%
# kommune_df['Dato'] < (np.datetime64('today')-np.timedelta64(2,'D'))

# %%
# Choose regions
kommunenavn = ["København"]
# kommunenavn = ["Horsens"]
# kommunenavn = ["København","Århus","Aalborg","Odense","Roskilde","Ishøj","Frederiksberg","Hvidovre","Greve","Rødovre","Skanderborg"]
kommunenavn = df.Kommunenavn.unique()

# Time stuff for plotting
df['Dato'] =  pd.to_datetime(df.Dato,format='%Y-%m-%d')


#Functions for loop
def antal_borgere(kommunenavn):
    return df_kommunekort["Antal borgere"][kommunenavn]
    # return df_kommunekort["Antal borgere"][kommunenavn]



for curKommune in kommunenavn:
    kommune_df = df.loc[df["Kommunenavn"] == curKommune]
    firstDate = np.datetime64(kommune_df.loc[kommune_df.index[0],'Dato'])-np.timedelta64(1,'D')
    firstDate = np.datetime64('2021-11-01')
    lastDate = np.datetime64(kommune_df.loc[kommune_df.index[-1],'Dato'])
    # Find number of citizens in region
    latestsubdir = list(os.walk(path_dash))[0][1][-1]
    latestdir = path_dash + latestsubdir
    df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                 delimiter = ';')
    df_kommunekort = df_kommunekort.set_index("Kommunenavn")
    
    kommune_nr = kommune_df.Kommune.iloc[0]
    kommune_df['Procent andel smittede'] = (kommune_df['Bekræftede tilfælde']/antal_borgere(curKommune))*100
    
    # Make figure

    fig,ax1 = plt.subplots(tight_layout=True,figsize=(10,6))

    # meanWidth=7
    # ax1.plot(kommune_df['Dato'],kommune_df['Procent andel smittede'],'k.:',linewidth=1,label=curKommune)
    # ax1.plot(rnTime(kommune_df['Dato'],meanWidth),rnMean(kommune_df['Procent andel smittede'],meanWidth),'k')
    # ax2 = ax1.twinx()
    # ax2.plot(kommune_df['Dato'],kommune_df['Bekræftede tilfælde'],'k.:',linewidth=1,label=curKommune)

    curDays = kommune_df['Dato'].values
    curPerc = kommune_df['Procent andel smittede'].values
    curCount = kommune_df['Bekræftede tilfælde'].values 

    # indexToUse = curDays <= (np.datetime64('today')-np.timedelta64(2,'D'))
    # curCount = curCount[indexToUse]
    # curPerc = curPerc[indexToUse]
    # curDays = curDays[indexToUse]
    indexToUse = curDays <= (np.datetime64(latestsubdir[-10:])-np.timedelta64(2,'D'))
    curCount = curCount[indexToUse]
    curPerc = curPerc[indexToUse]
    curDays = curDays[indexToUse]

    meanWidth = 7

    ax1.plot(curDays,curPerc,'k.:',linewidth=1,label=curKommune)
    ax1.plot(rnTime(curDays,meanWidth),rnMean(curPerc,meanWidth),'k')
    ax2 = ax1.twinx()
    ax2.plot(curDays,curCount,'k.:',linewidth=1,label=curKommune)


    ax1.set_title(curKommune)

    # ax1.legend(loc='upper left')
    # ax1.grid()
    ax1.set_ylabel('Procent smittede [%]')
    ax2.set_ylabel('Antal smittede')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_xlim([firstDate,lastDate])
    ax1.set_xlim([firstDate,lastDate+np.timedelta64(7,'D')])
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    
    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
         curSunday = firstSunday + np.timedelta64(7*k,'D')
         ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    ax1.grid(axis='y')

    #Tilts the x labels. 
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    if saveFigures:
        fig.savefig(path_figs+'Kommune/'+curKommune)

# %%
# df_kommunekort

# antal_borgere(kommunenavn[i])
# kommunenavn[4]

plt.figure()

plt.plot(curDays,curCount)
plt.plot(rnTime(curDays,7),rnMean(curCount,7))

plt.xlim(left=np.datetime64('2021-11-01'))

# %%
# rnMean(curCount,7)[-10:]
curMean = rnMean(curCount,7)

np.diff(curMean)[-10:]

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(12,8))

ax1.plot(curDays[6:],curMean)
# ax2.plot(curDays[7:],np.sign(np.diff(curMean)),'*--')
ax2.plot(curDays[7:],np.diff(curMean),'*--')
ax1.set_xlim(left=np.datetime64('2021-11-01'))
ax2.grid()

# %%



