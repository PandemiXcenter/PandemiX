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
# Define paths
rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 

path_data = rootdir_data + "ssi_data\\"
path_dash = rootdir_data + "ssi_dashboard\\"
path_vacc = rootdir_data + "ssi_vacc\\"
path_figs = os.getcwd() +"\\..\\Figures\\" 

path_figs = path_figs+"KommuneAlder\\"

# %%
# Find number of citizens in region
latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                delimiter = ';')
df_kommunekort = df_kommunekort.set_index("Kommunenavn")

# %% [markdown]
# # Population size
# Populationnumbers have been downloaded from Danmark Statistik.
# 
# File are in two separate files. First, combine into one dataframe

# %%
popdf1 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del1.csv',header=None,encoding='latin1',delimiter=';')
popdf2 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del2.csv',header=None,encoding='latin1',delimiter=';')

popdf = pd.concat([popdf1,popdf2])

popdf = popdf.rename(columns={0:"Kommune",1:'Alder',2:'Antal'})
popdf['AlderKort'] = popdf.Alder.apply(lambda x: int(str(x).split(' ')[0]))
totCounts = popdf.groupby('Kommune').sum()
popdf.head()

# %%
def getPopSize(kommuneNavn,minAlder=0,maxAlder=125):

    if (kommuneNavn == 'Høje Tåstrup'):
        kommuneNavn = 'Høje-Taastrup'
    if (kommuneNavn == 'Århus'):
        kommuneNavn = 'Aarhus'
    if (kommuneNavn == 'Nordfyn'):
        kommuneNavn = 'Nordfyns'
    if (kommuneNavn == 'Vesthimmerland'):
        kommuneNavn = 'Vesthimmerlands'

        
    return popdf[(popdf.Kommune == kommuneNavn) & (popdf.AlderKort >= minAlder) & (popdf.AlderKort <= maxAlder)].Antal.sum()
    
# kommuneNavn = 'København'
# minAlder = 0
# maxAlder = 2

# getPopSize('København',0,20)

# %%
getPopSize('Århus')
# np.sort(popdf.Kommune.unique())

# %% [markdown]
# # Get data
# 
# Since data only contains the most recent numbers, go through every directory to generate time-series

# %%
allSubDirs = list(os.walk(path_dash))[0][1]

df = pd.DataFrame()

for curSubDir in allSubDirs:
    curdir = path_dash + curSubDir
    curfilepath = curdir+'/Kommunalt_DB/17_tilfaelde_fnkt_alder_kommuner.csv'

    # Check if file was included at the time. The "Kommune/17" file wasn't included until 2021-09-22 
    if os.path.isfile(curfilepath):
        curdf = pd.read_csv(curfilepath,encoding='latin1',delimiter = ';')
        
        df = pd.concat([df,curdf])

# Set dtypes
df.Kommune = df.Kommune.fillna(0)  # All NaN kommuner is set to zero
df['Kommune'] = df['Kommune'].astype(int)
df['Dagsdato'] = pd.to_datetime(df['Dagsdato'])
df['Bekræftede tilfælde'] = pd.to_numeric(df['Bekræftede tilfælde'])
df['Aldersgruppe'] = df.Aldersgruppe.replace('00-02','0-2')
df['Aldersgruppe'] = df.Aldersgruppe.replace('03-05','3-5')
df['Aldersgruppe'] = df.Aldersgruppe.replace('06-11','6-11')

# df['Forskel'] = df['Bekræftede tilfælde'].diff().fillna(0).astype(int,errors='ignore')

# %%
# np.array(dayDiff).unique()
# pd.DataFrame(data=dayDiff).loc[:,0].unique()

# df['Kommune']

# %%

df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                delimiter = ';')
df_kommunekort = df_kommunekort.set_index("Kommunenavn")

# kommune_nr = kommune_df.Kommune.iloc[0]
# df_kommunekort['København'] 
df_kommunekort['Kommune']['København']

# %%


# %%

# curKom = 101
# curAge = '6-11'
# curdf = df[(df.Kommune == curKom) & (df.Aldersgruppe == curAge)]
# curdf.tail(10)
# # df[df.Kommune == 101].tail(20)

def getDiffTimeSeries(komCode,Age):

    curdf = df[(df.Kommune == komCode) & (df.Aldersgruppe == Age)] 

    if (len(curdf) == 0):
        return np.array(np.datetime64('today')),np.array([0])
        
    dayDiff = [int(x) for x in (curdf.Dagsdato.diff()/np.timedelta64(1,'D')).fillna(0)]

    curDays = []
    curDiffs = []
    for i in range(1,len(curdf)):
        curRow = curdf.iloc[i]
        prevRow = curdf.iloc[i-1]
        # print(curRow)
        if (dayDiff[i] == 1):
            curDays.append(curRow.Dagsdato)
            curDiffs.append(curRow['Bekræftede tilfælde']-prevRow['Bekræftede tilfælde']) 
        elif (dayDiff[i] == 3):
            curCount = curRow['Bekræftede tilfælde']-prevRow['Bekræftede tilfælde']
            curDays.append(curRow.Dagsdato-np.timedelta64(2,'D'))
            curDays.append(curRow.Dagsdato-np.timedelta64(1,'D'))
            curDays.append(curRow.Dagsdato)
            curDiffs.append(curCount/3) 
            curDiffs.append(curCount/3) 
            curDiffs.append(curCount/3) 

    return np.array(curDays),np.array(curDiffs)

curDates,curCounts = getDiffTimeSeries(101,'6-11')
# df_kommunekort['Kommune']['Samsø']
curDates,curCounts = getDiffTimeSeries(741,'0-2')
# getDiffTimeSeries(741,'3-5')
# print(curDates[-10:])
# print(curCounts[-10:])
# plt.figure()
# plt.plot(curDays,curDiffs)
# plt.plot(rnTime(curDays,7),rnMean(curDiffs,7))

# %%
# asdf = pd.DataFrame() 
# asdf['Kommunenavn'] = df_kommunekort.index.unique()
# asdf['Samlet population'] = [getPopSize(x) for x in df_kommunekort.index.unique()]
# asdf

# %%
allAge = df.Aldersgruppe.unique()[:-1]

import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue","green"],len(allAge))
cmap = plt.cm.get_cmap('turbo',len(allAge))

firstDate = np.datetime64('2021-12-01')

curKom = 851
curKom = 101
curKommuneNavn = 'København'

posKommuneNavn = df_kommunekort.index.values[1:]
# posKommuneNavn = ['Høje Tåstrup','Århus','Nordfyn','Vesthimmerland']
# posKommuneNavn = ['Samsø']

for curKommuneNavn in posKommuneNavn:

    # try:
    fig,ax1 = plt.subplots()

    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')

    curKom = df_kommunekort['Kommune'][curKommuneNavn]
    # for curAge in allAge:
    for i in range(0,len(allAge)):
        curAge = allAge[i]
        curColor = cmap(i)
        curDates,curCounts = getDiffTimeSeries(curKom,curAge)
        # ax1.plot(curDates,curCounts,'.:',linewidth=0.5,markersize=2,color=curColor)
        # ax1.plot(rnTime(curDates,7),rnMean(curCounts,7),label=curAge,color=curColor)

        if (len(curCounts) > 7):
            ax1.plot(curDates[6:],rnMean(curCounts,7),label=curAge,color=curColor)

    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

    ax1.set_xlim()
    ax1.set_xlim(left=firstDate)

    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
            curSunday = firstSunday + np.timedelta64(7*k,'D')
            ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    ax1.grid(axis='y')
    ax1.legend()

    # ax1.set_title(curKom)
    # ax1.set_title('København')
    ax1.set_ylabel('Smittetilfælde, 7-dages gennemsnit')
    ax1.set_title(curKommuneNavn)

    if saveFigures:
        fig.savefig(path_figs+curKommuneNavn+'_Antal')

    plt.close('all')
    # except:
    #     2+2

# %%
# # df.groupby(['Kommune','Aldersgruppe'])
# # asdf = df.set_index(['Kommune','Aldersgruppe'])
# df.columns

# curKom = 101
# curAge = '6-11'
# asdf = df[(df.Kommune == curKom) & (df.Aldersgruppe == curAge)]

# plt.figure()
# plt.plot(asdf.Dagsdato,asdf['Bekræftede tilfælde'].diff(),'*')
# plt.plot(rnTime(asdf.Dagsdato,7),rnMean(asdf['Bekræftede tilfælde'].diff(),7))

# %% [markdown]
# # Normed by population size

# %%


curKom = 851
curKom = 101
curKommuneNavn = 'København'

posKommuneNavn = df_kommunekort.index.values[1:]
# posKommuneNavn = ['Høje Tåstrup','Århus','Nordfyn','Vesthimmerland']
# posKommuneNavn = ['Samsø']

for curKommuneNavn in posKommuneNavn:

    # try: 
    fig,ax1 = plt.subplots()
    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')

    
    for i in range(0,len(allAge)):
        curAge = allAge[i]
        if (curAge == '80+'):
            curMinAge,curMaxAge = 80,125
        else:
            curMinAge,curMaxAge = [int(x) for x in curAge.split('-')]

        curColor = cmap(i)
        curKom = df_kommunekort['Kommune'][curKommuneNavn]
        curDates,curCounts = getDiffTimeSeries(curKom,curAge)
        # ax1.plot(curDates,curCounts,'.:',linewidth=0.5,markersize=2,color=curColor)
        # ax1.plot(rnTime(curDates,7),rnMean(curCounts,7),label=curAge,color=curColor)
        # ax1.plot(curDates[6:],rnMean(curCounts,7),label=curAge,color=curColor)

        
        curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)

        if (len(curCounts) > 7):
            # ax1.plot(curDates,100*curCounts/curPopSize,'.:',linewidth=0.5,markersize=2,color=curColor)
            ax1.plot(curDates[6:],100*rnMean(curCounts,7)/curPopSize,label=curAge,color=curColor)

    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

    ax1.set_xlim()
    ax1.set_xlim(left=firstDate)
    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
            curSunday = firstSunday + np.timedelta64(7*k,'D')
            ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    ax1.grid(axis='y')
    ax1.legend()
    ax1.set_ylabel('Smittetilfælde, som andel af befolkning [%]\n7-dages gennemsnit')
    ax1.set_title(curKommuneNavn)

    if saveFigures:
        fig.savefig(path_figs+curKommuneNavn+'_Procent')
    # except:
    #     2+2

# %%
plt.close('all')

# %%

posKommuneNavn = df_kommunekort.index.values[1:]
# posKommuneNavn = ['Høje Tåstrup','Århus','Nordfyn','Vesthimmerland']


maxCumu = 0

maxKommune = ''

for curKommuneNavn in posKommuneNavn:
        for i in range(0,len(allAge)):
            curAge = allAge[i]
            if (curAge == '80+'):
                curMinAge,curMaxAge = 80,125
            else:
                curMinAge,curMaxAge = [int(x) for x in curAge.split('-')]

            curColor = cmap(i)
            curKom = df_kommunekort['Kommune'][curKommuneNavn]
            curDates,curCounts = getDiffTimeSeries(curKom,curAge)
            
            curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)

            if (len(curCounts) > 7):
                curCumu = 100*np.cumsum(curCounts)/curPopSize

            curMax = np.max(curCumu)

            if (curMax > maxCumu):
                maxKommune = curKommuneNavn
                maxCumu = curMax 
            # maxCumu = np.max([curMax,maxCumu])
print(f'{maxKommune}: {maxCumu}')


# %%


# %%
# fig,ax1 = plt.subplots()

# fig.patch.set_facecolor('xkcd:off white')
# ax1.set_facecolor('xkcd:off white')

# curKom = 851
# curKom = 101
# curKommuneNavn = 'København'


posKommuneNavn = df_kommunekort.index.values[1:]
# posKommuneNavn = ['Høje Tåstrup','Århus','Nordfyn','Vesthimmerland']
# posKommuneNavn = ['Hvidovre']
# posKommuneNavn = ['Samsø']

for curKommuneNavn in posKommuneNavn:

    # try: 
    plt.close('all')
    fig,ax1 = plt.subplots()
    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')

    for i in range(0,len(allAge)):
        curAge = allAge[i]
        if (curAge == '80+'):
            curMinAge,curMaxAge = 80,125
        else:
            curMinAge,curMaxAge = [int(x) for x in curAge.split('-')]

        curColor = cmap(i)
        curKom = df_kommunekort['Kommune'][curKommuneNavn]
        curDates,curCounts = getDiffTimeSeries(curKom,curAge)
        # ax1.plot(curDates,curCounts,'.:',linewidth=0.5,markersize=2,color=curColor)
        # ax1.plot(rnTime(curDates,7),rnMean(curCounts,7),label=curAge,color=curColor)
        # ax1.plot(curDates[6:],rnMean(curCounts,7),label=curAge,color=curColor)

        
        curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)

        if (len(curCounts) > 7):
            # Cutoff everything before first date
            curDates = np.array([np.datetime64(x) for x in curDates])
            curIndex = (curDates > firstDate)
            curDates = curDates[curIndex]
            curCounts = curCounts[curIndex]

            # ax1.plot(curDates,100*curCounts/curPopSize,'.:',linewidth=0.5,markersize=2,color=curColor)
            # ax1.plot(curDates[6:],100*rnMean(np.cumsum(curCounts),7)/curPopSize,label=curAge,color=curColor)
            ax1.plot(curDates,100*np.cumsum(curCounts)/curPopSize,label=curAge,color=curColor)

    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

    ax1.set_xlim()
    ax1.set_xlim(left=firstDate)
    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
            curSunday = firstSunday + np.timedelta64(7*k,'D')
            ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    ax1.grid(axis='y')
    ax1.legend()
    # ax1.set_ylabel('Kumuleret sum af smittetilfælde, 7-dages gennemsnit\nAndel af befolkning [%]')
    # ax1.set_ylabel('Kumuleret sum af smittetilfælde\nAndel af befolkning [%]')
    ax1.set_ylabel(f'Kumuleret sum af smittetilfælde siden 1. november\nAndel af befolkning [%]')
    # ax1.set_ylabel(f'Kumuleret sum af smittetilfælde siden 1. december\nAndel af befolkning [%]')
    ax1.set_title(curKommuneNavn)

    if saveFigures:
        fig.savefig(path_figs+curKommuneNavn+'_Kumuleret')
    # except:
    #     print(f'Could not plot {curKommuneNavn}')

# %%
plt.close('all')

# %% [markdown]
# # Immunity overview

# %%
# fig,ax1 = plt.subplots()

# fig.patch.set_facecolor('xkcd:off white')
# ax1.set_facecolor('xkcd:off white')

# curKom = 851
# curKom = 101
# curKommuneNavn = 'København'


posKommuneNavn = df_kommunekort.index.values[1:]
# posKommuneNavn = ['Høje Tåstrup','Århus','Nordfyn','Vesthimmerland']
# posKommuneNavn = ['Hvidovre']
# posKommuneNavn = ['Samsø']

for curKommuneNavn in posKommuneNavn:

    # try: 
    plt.close('all')
    fig,ax1 = plt.subplots()
    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')


    curPopSizeFull = getPopSize(curKommuneNavn)
    curTotCaseCount = 0
    curBarVals = []
    allAgesToPlot = []
    for i in range(0,len(allAge)):
        curAge = allAge[i]
        if (curAge == '80+'):
            curMinAge,curMaxAge = 80,125
        else:
            curMinAge,curMaxAge = [int(x) for x in curAge.split('-')]

        curColor = cmap(i)
        curKom = df_kommunekort['Kommune'][curKommuneNavn]
        curDates,curCounts = getDiffTimeSeries(curKom,curAge)
        # ax1.plot(curDates,curCounts,'.:',linewidth=0.5,markersize=2,color=curColor)
        # ax1.plot(rnTime(curDates,7),rnMean(curCounts,7),label=curAge,color=curColor)
        # ax1.plot(curDates[6:],rnMean(curCounts,7),label=curAge,color=curColor)

        
        curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)

        if (len(curCounts) > 7):
            # Cutoff everything before first date
            curDates = np.array([np.datetime64(x) for x in curDates])
            curIndex = (curDates > firstDate)
            curDates = curDates[curIndex]
            curCounts = curCounts[curIndex]

            # ax1.plot(curDates,100*curCounts/curPopSize,'.:',linewidth=0.5,markersize=2,color=curColor)
            # ax1.plot(curDates[6:],100*rnMean(np.cumsum(curCounts),7)/curPopSize,label=curAge,color=curColor)

            curTotCaseCount = curTotCaseCount + np.cumsum(curCounts)[-1]
            natImmu = 100*np.cumsum(curCounts)/curPopSize
            curBarVals.append(natImmu[-1])
            allAgesToPlot.append(curAge)
            # ax1.plot(curDates,100*np.cumsum(curCounts)/curPopSize,label=curAge,color=curColor)


    # ax1.bar(allAge,100*np.ones(np.array(curBarVals).shape),color='gray')
    # ax1.bar(allAge,curBarVals,color='xkcd:dark green')
    # agesToPlot = np.concatenate([['Samlet'],allAge])
    agesToPlot = np.concatenate([['Samlet'],allAgesToPlot])
    valsToPlot = np.concatenate([[100*curTotCaseCount/curPopSizeFull],curBarVals])
    ax1.bar(agesToPlot,100*np.ones(np.array(valsToPlot).shape),color='gray')
    ax1.bar(agesToPlot,valsToPlot,color='xkcd:dark green')

    ax1.plot([0.5,0.5],[0,100],'k--')

    ax1.set_ylim([0,100])

    ax1.set_xlabel('Aldersgruppe')
    ax1.set_ylabel('Andel af borgere smittet siden 1. december [%]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.set_yticks(np.arange(0,101,10))
    ax1.grid(axis='y')
    
    ax1.set_title(curKommuneNavn)

    if saveFigures:
        fig.savefig(path_figs+curKommuneNavn+'_NaturligImmunitet')

# %%
agesToPlot = np.concatenate([allAge,['Samlet']])
np.array(valsToPlot).shape 
# agesToPlot
# ax1.bar(agesToPlot,100*np.ones(np.array(valsToPlot).shape),color='gray')
[100*curTotCaseCount/curPopSizeFull]

# %% [markdown]
# # Generate markdown for web

# %%

posKommuneNavn = np.sort(df_kommunekort.index.values[1:])

for curKommuneNavn in posKommuneNavn:

    print(f'![](../Figures/KommuneAlder/{curKommuneNavn}_Antal.png) | ![](../Figures/KommuneAlder/{curKommuneNavn}_Procent.png) | ![](../Figures/KommuneAlder/{curKommuneNavn}_Kumuleret.png)')



