
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

# %%
latestsubdir = list(os.walk(path_data))[0][1][-1]
latestdir = path_data + latestsubdir

dfCase = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
dfCase = dfCase.iloc[:-2]
dfCase['NewPositive'] = pd.to_numeric(dfCase['NewPositive'].astype(str).apply(lambda x: x.replace('.','')))
dfCase['Tested'] = pd.to_numeric(dfCase['Tested'].astype(str).apply(lambda x: x.replace('.','')))
dfCase['PosPct'] = pd.to_numeric(dfCase['PosPct'].astype(str).apply(lambda x: x.replace(',','.')))
dfCase['Date'] =  pd.to_datetime(dfCase.Date,format='%Y-%m-%d')
testDates = dfCase['Date']

dfAdm = pd.read_csv(latestdir+'/Newly_admitted_over_time.csv',delimiter = ';',dtype=str)
dfAdm['Dato'] = pd.to_datetime(dfAdm['Dato'])
dfAdm['Total'] = pd.to_numeric(dfAdm['Total'])
dfAdm.tail()


dfDea = pd.read_csv(latestdir+'/Deaths_over_time.csv',delimiter = ';',dtype=str)
dfDea = dfDea.iloc[:-1,:]
dfDea['Dato'] = pd.to_datetime(dfDea['Dato'])
dfDea['Antal_døde'] = pd.to_numeric(dfDea['Antal_døde'])
dfDea.tail()


# %%
allDates = dfCase.Date
allDatesShift = allDates + np.timedelta64(365,'D')

allDatesAdm = dfAdm.Dato
allDatesAdmShift = allDatesAdm + np.timedelta64(365,'D')

allDatesDea = dfDea.Dato
allDatesDeaShift = allDatesDea + np.timedelta64(365,'D')


firstDate = np.datetime64('2021-10-01')-np.timedelta64(1,'D')
# lastDate = np.datetime64('2022-03-01')+np.timedelta64(1,'D')
lastDate = np.datetime64('2022-03-01')

meanWidth = 7

# %% [markdown]
# # Cases

# %%

allCases = dfCase.NewPositive.values

fig,ax1 = plt.subplots(tight_layout=True)


ax1.plot(allDatesShift[:-1],allCases[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(allDatesShift[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(allDates[:-1],allCases[:-1],'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(allDates[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


# ax1.plot(allDatesShift,allCases,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift,meanWidth),rnMean(allCases,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
# ax1.plot(allDates,allCases,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
# ax1.plot(rnTime(allDates,meanWidth),rnMean(allCases,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')

ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal tilfælde')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Positive')

maxAvg = np.max(rnMean(allCases,meanWidth))
ax1.set_ylim(top=maxAvg*1.1)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Positive_zoom')

# %% [markdown]
# # Indlæggelser

# %%

allAdms = dfAdm.Total.values

fig,ax1 = plt.subplots(tight_layout=True)


ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(allDatesAdm,allAdms,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(allDatesAdm,meanWidth),rnMean(allAdms,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal nyindlæggelser')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Nyindlagte')

maxAvg = np.max(rnMean(allAdms,meanWidth))
ax1.set_ylim(top=maxAvg*1.1)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Nyindlagte_zoom')

# %% [markdown]
# # Dødsfald

# %%

allDeas = dfDea['Antal_døde'].values

fig,ax1 = plt.subplots(tight_layout=True)


ax1.plot(allDatesDeaShift[:-1],allDeas[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(allDatesDeaShift[:-1],meanWidth),rnMean(allDeas[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(allDatesDea[:-1],allDeas[:-1],'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(allDatesDea[:-1],meanWidth),rnMean(allDeas[:-1],meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal dødsfald')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Doedsfald')

maxAvg = np.max(rnMean(allDeas,meanWidth))
ax1.set_ylim(top=maxAvg*1.1)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Doedsfald_zoom')

# %% [markdown]
# # Indlæggelser

# %%

latestsubdir_dash = list(os.walk(path_dash))[0][1]
# latestsubdir_dash == 'SSI_dashboard_2021-12-17'
lastFullFileIndex = np.where([x == 'SSI_dashboard_2021-12-17' for x in latestsubdir_dash])[0][0]

# %%
# Until 2021-12-20, all dates were included in one file. Since then, additional data was added, and the file only contains the most recent numbers

# latestsubdir_dash = list(os.walk(path_dash))[0][1][-11]
# latestsubdir_dash = 'SSI_dashboard_2021-12-17'

latestsubdirs_dash = list(os.walk(path_dash))[0][1]
# latestsubdirs_dash == 'SSI_dashboard_2021-12-17'
lastFullFileIndex = np.where([x == 'SSI_dashboard_2021-12-17' for x in latestsubdirs_dash])[0][0]
latestdir_dash = path_dash + latestsubdirs_dash[lastFullFileIndex]

dfKey = pd.read_csv(latestdir_dash+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')
dfKey

dfKeysArray = []
for k in range(lastFullFileIndex+1,len(latestsubdirs_dash)):
    
    latestdir_dash = path_dash + latestsubdirs_dash[k]
    curdf = pd.read_csv(latestdir_dash+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')
    dfKeysArray.append(curdf)
# dfKeysArray

dfKey['IndberetningDato'] = pd.to_datetime(dfKey['IndberetningDato'])

# latestdir_dash

# %%
print(dfKey.columns)
print(dfKeysArray[-1].columns)


# %%
# Make arrays to plot
keyDates = dfKey.IndberetningDato
keyDatesShift = keyDates + np.timedelta64(365,'D')
keyAdm = dfKey['Antal indlagte i dag med COVID']
keyAdmInt = dfKey['Antal indlagt i dag på intensiv']
keyAdmResp = dfKey['Antal indlagt i dag og i respirator']

## Add the new data

# 2021-12-20 still used old names
dateToAdd = np.datetime64(pd.to_datetime(dfKeysArray[0].IndberetningDato.values[0]))
keyDates = np.append(keyDates,dateToAdd)
keyAdm = np.append(keyAdm,dfKeysArray[0]['Antal indlagte i dag med COVID'][0])
keyAdmInt = np.append(keyAdmInt,dfKeysArray[0]['Antal indlagt i dag på intensiv'][0])
keyAdmResp = np.append(keyAdmResp,dfKeysArray[0]['Antal indlagt i dag og i respirator'][0])




for k in range(1,len(dfKeysArray)):
    thisDate = dfKeysArray[k].Dato[0]
    thisAdm = dfKeysArray[k]['Indlæggelser i dag'][0]
    thisAdmInt = dfKeysArray[k]['Indlæggelser i dag (intensiv)'][0]
    thisAdmResp = dfKeysArray[k]['Indlæggelser i dag (respirator)'][0]
    # # print(dfKeysArray[k])
    # print(thisDate)
    # print(thisAdm)
    # print(thisAdmInt)

    keyDates = np.append(keyDates,np.datetime64(thisDate))
    keyAdm = np.append(keyAdm,thisAdm)
    keyAdmInt = np.append(keyAdmInt,thisAdmInt)
    keyAdmResp = np.append(keyAdmResp,thisAdmResp)



keyDates = keyDates.astype('datetime64[D]')
keyDatesShift = keyDates + np.timedelta64(365,'D')

# %%
fig,ax1 = plt.subplots()

ax1.plot(keyDatesShift,keyAdm,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdm,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(keyDates,keyAdm,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(keyDates,meanWidth),rnMean(keyAdm,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal indlagte')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_Indlagte')


# %%
fig,ax1 = plt.subplots()

ax1.plot(keyDatesShift,keyAdmInt,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdmInt,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(keyDates,keyAdmInt,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(keyDates,meanWidth),rnMean(keyAdmInt,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal indlagte på intensiv')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_IndlagteIntensiv')


# %%
fig,ax1 = plt.subplots()

ax1.plot(keyDatesShift,keyAdmResp,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdmResp,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(keyDates,keyAdmResp,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
ax1.plot(rnTime(keyDates,meanWidth),rnMean(keyAdmResp,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('Antal indlagte i respirator')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


ax1.set_ylim(top=100)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_IndlagteRespirator')


# %%
fig,(ax1,ax2) = plt.subplots(2,1)

ax1.plot(keyDatesShift,keyAdm,'k.:',markersize=4,linewidth=0.5,label='Indlagte')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdm,meanWidth),'k',label=f'Indlagte, {meanWidth} dages gennemsnit')
ax1.plot(keyDatesShift,keyAdmInt,'m.:',markersize=4,linewidth=0.5,label='Intensiv')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdmInt,meanWidth),'m',label=f'Intensiv, {meanWidth} dages gennemsnit')
ax1.plot(keyDatesShift,keyAdmResp,'r.:',markersize=4,linewidth=0.5,label='Respirator')
ax1.plot(rnTime(keyDatesShift,meanWidth),rnMean(keyAdmResp,meanWidth),'r',label=f'Respirator, {meanWidth} dages gennemsnit')

ax2.plot(keyDates,keyAdm,'k.:',markersize=4,linewidth=0.5,label='Indlagte')
ax2.plot(rnTime(keyDates,meanWidth),rnMean(keyAdm,meanWidth),'k',label=f'Indlagte, {meanWidth} dages gennemsnit')
ax2.plot(keyDates,keyAdmInt,'m.:',markersize=4,linewidth=0.5,label='Intensiv')
ax2.plot(rnTime(keyDates,meanWidth),rnMean(keyAdmInt,meanWidth),'m',label=f'Intensiv, {meanWidth} dages gennemsnit')
ax2.plot(keyDates,keyAdmResp,'r.:',markersize=4,linewidth=0.5,label='Respirator')
ax2.plot(rnTime(keyDates,meanWidth),rnMean(keyAdmResp,meanWidth),'r',label=f'Respirator, {meanWidth} dages gennemsnit')


ax1.legend(loc='upper left',fontsize=12)
ax1.grid()
ax1.set_ylabel('Antal personer')
ax1.set_ylim(bottom=0)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


# ax2.legend(loc='upper left')
ax2.grid()
ax2.set_ylabel('Antal personer')
ax2.set_ylim(bottom=0)
ax2.set_xlim([firstDate,lastDate])
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax2.spines['top'].set_visible(False) 
ax2.spines['right'].set_visible(False)

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_IndlagteAll')


# %%
np.max([100,np.max(keyAdmResp)*1.1])

# latestsubdir_dash = list(os.walk(path_dash))[0][1][-1]
# # latestsubdir_dash = 'SSI_dashboard_2021-12-17'
# latestdir_dash = path_dash + latestsubdir_dash

# dfKey = pd.read_csv(latestdir_dash+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')
# dfKey

# # curSubDir = list(os.walk(path_dash))[0][1][0]

# # thisPath = path_dash + curSubDir

# # curdf = pd.read_csv(thisPath+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')

# # curdf
# # # print(thisPath+'\\Kommunalt_DB\\01_noegletal.csv')
# # # "D:\\Pandemix\\Github\\DanskeData\\ssi_dashboard\\SSI_dashboard_2021-03-08\\Kommunalt_DB\\01_noegletal.csv"

# %% [markdown]
# # Compare cases and admissions

# %%


# %%

allCases = dfCase.NewPositive.values

fig,ax1 = plt.subplots(tight_layout=True)

cases2021 = allCases[allDates < np.datetime64('2021-03-01')]
max2021Cases = np.max(rnMean(cases2021,meanWidth))

# ax1.plot(allDatesShift[:-1],allCases[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')


ax1.plot(allDates[:-1],100*allCases[:-1]/max2021Cases,'b.:',markersize=4,linewidth=0.5,label='Smittetilfælde')
ax1.plot(rnTime(allDates[:-1],meanWidth),100*rnMean(allCases[:-1],meanWidth)/max2021Cases,'b',label=f'Smittetilfælde, {meanWidth} dages gennemsnit')

allAdms = dfAdm.Total.values
allAdms = dfAdm.Total
Adms2021 = allAdms[allDates < np.datetime64('2021-03-01')]
max2021Adms = np.max(rnMean(Adms2021,meanWidth))


# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(allDatesAdm,meanWidth),100*rnMean(allAdms,meanWidth)/max2021Adms,'r',label=f'Indlæggelser, {meanWidth} dages gennemsnit')



# ax1.plot(allDatesShift,allCases,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift,meanWidth),rnMean(allCases,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
# ax1.plot(allDates,allCases,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
# ax1.plot(rnTime(allDates,meanWidth),rnMean(allCases,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')

ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('% af maximal værdi vinteren 2020/2021')
ax1.set_ylim(bottom=0)
firstDate = np.datetime64('2020-11-01')
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_CasesHospCompare')

# maxAvg = np.max(rnMean(allCases,meanWidth))
# ax1.set_ylim(top=maxAvg*1.1)

firstDate = np.datetime64('2021-10-01')
ax1.set_xlim([firstDate,lastDate])

if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_CasesHospCompare_zoom')

# %%

allCases = dfCase.NewPositive.values

fig,ax1 = plt.subplots(tight_layout=True)

cases2021 = allCases[allDates < np.datetime64('2021-03-01')]
max2021Cases = np.max(rnMean(cases2021,meanWidth))

# ax1.plot(allDatesShift[:-1],allCases[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')


# ax1.plot(allDates[:-1],100*allCases[:-1]/max2021Cases,'b.:',markersize=4,linewidth=0.5,label='Smittetilfælde')
ax1.plot(rnTime(allDates[:-1],meanWidth),100*rnMean(allCases[:-1],meanWidth)/max2021Cases,'b',label=f'Smittetilfælde, {meanWidth} dages gennemsnit')

allAdms = dfAdm.Total.values
allAdms = dfAdm.Total
Adms2021 = allAdms[allDates < np.datetime64('2021-03-01')]
max2021Adms = np.max(rnMean(Adms2021,meanWidth))


# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(allDatesAdm,meanWidth),100*rnMean(allAdms,meanWidth)/max2021Adms,'xkcd:orange',label=f'Indlæggelser, {meanWidth} dages gennemsnit')


# allAdms = keyAdmInt.Total.values
allAdmsInt = keyAdmInt
Adms2021Int = allAdmsInt[keyDates < np.datetime64('2021-03-01')]
max2021AdmsInts = np.max(rnMean(keyAdmInt,meanWidth))

# keyAdmInt
# keyDates

# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(keyDates,meanWidth),100*rnMean(allAdmsInt,meanWidth)/max2021AdmsInts,'r',label=f'Intensiv, {meanWidth} dages gennemsnit')


# ax1.plot(allDatesShift,allCases,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift,meanWidth),rnMean(allCases,meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
# ax1.plot(allDates,allCases,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
# ax1.plot(rnTime(allDates,meanWidth),rnMean(allCases,meanWidth),'b',label=f'2021/2022, {meanWidth} dages gennemsnit')

ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('% af maximal værdi vinteren 2020/2021')
ax1.set_ylim(bottom=0)
firstDate = np.datetime64('2020-11-01')
curXticks = np.arange(np.datetime64('2020-11'),np.datetime64('2022-11'))
ax1.set_xticks(curXticks)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_CasesHospCompare2')

# maxAvg = np.max(rnMean(allCases,meanWidth))
# ax1.set_ylim(top=maxAvg*1.1)

firstDate = np.datetime64('2021-10-01')
ax1.set_xlim([firstDate,lastDate])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
if saveFigures:
    plt.savefig(path_figs+'VinterSammenligning_CasesHospCompare2_zoom')

# %%

allCases = dfCase.NewPositive.values

fig,ax1 = plt.subplots(tight_layout=True)

cases2021 = allCases[allDates < np.datetime64('2021-03-01')]
max2021Cases = np.max(rnMean(cases2021,meanWidth))

# ax1.plot(allDatesShift[:-1],allCases[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')


# ax1.plot(allDates[:-1],100*allCases[:-1]/max2021Cases,'b.:',markersize=4,linewidth=0.5,label='Smittetilfælde')
ax1.plot(rnTime(allDates[:-1],meanWidth),100*rnMean(allCases[:-1],meanWidth)/max2021Cases,'b',label=f'Cases, {meanWidth} day running mean')

allAdms = dfAdm.Total.values
allAdms = dfAdm.Total
Adms2021 = allAdms[allDates < np.datetime64('2021-03-01')]
max2021Adms = np.max(rnMean(Adms2021,meanWidth))


# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(allDatesAdm,meanWidth),100*rnMean(allAdms,meanWidth)/max2021Adms,'xkcd:orange',label=f'Hospital admissions, {meanWidth} day running mean')


# allAdms = keyAdmInt.Total.values
allAdmsInt = keyAdmInt
Adms2021Int = allAdmsInt[keyDates < np.datetime64('2021-03-01')]
max2021AdmsInts = np.max(rnMean(keyAdmInt,meanWidth))

# keyAdmInt
# keyDates

# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(keyDates,meanWidth),100*rnMean(allAdmsInt,meanWidth)/max2021AdmsInts,'r',label=f'ICU admissions, {meanWidth} day running mean')


# ax1.plot(allDatesShift,allCases,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift,meanWidth),rnMean(allCases,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDates,allCases,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
# ax1.plot(rnTime(allDates,meanWidth),rnMean(allCases,meanWidth),'b',label=f'2021/2022, {meanWidth} day running mean')

ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('% of maximal value in winter 2020/2021')
ax1.set_ylim(bottom=0)
firstDate = np.datetime64('2020-11-01')
curXticks = np.arange(np.datetime64('2020-11'),np.datetime64('2022-11'))
ax1.set_xticks(curXticks)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


if saveFigures:
    plt.savefig(path_figs+'CasesHospCompare_English')

# maxAvg = np.max(rnMean(allCases,meanWidth))
# ax1.set_ylim(top=maxAvg*1.1)

firstDate = np.datetime64('2021-10-01')
ax1.set_xlim([firstDate,lastDate])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
if saveFigures:
    plt.savefig(path_figs+'CasesHospCompare_English_zoom')

# %%


# %%

allCases = dfCase.NewPositive.values

fig,ax1 = plt.subplots(tight_layout=True)

cases2021 = allCases[allDates < np.datetime64('2021-03-01')]
max2021Cases = np.max(rnMean(cases2021,meanWidth))

# ax1.plot(allDatesShift[:-1],allCases[:-1],'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift[:-1],meanWidth),rnMean(allCases[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')


# ax1.plot(allDates[:-1],100*allCases[:-1]/max2021Cases,'b.:',markersize=4,linewidth=0.5,label='Smittetilfælde')
ax1.plot(rnTime(allDates[:-1],meanWidth),100*rnMean(allCases[:-1],meanWidth)/max2021Cases,'b',label=f'Cases, {meanWidth} day running mean')

allAdms = dfAdm.Total.values
allAdms = dfAdm.Total
Adms2021 = allAdms[allDates < np.datetime64('2021-03-01')]
max2021Adms = np.max(rnMean(Adms2021,meanWidth))


# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(allDatesAdm,meanWidth),100*rnMean(allAdms,meanWidth)/max2021Adms,'xkcd:orange',label=f'Hospital admissions, {meanWidth} day running mean')


# allAdms = keyAdmInt.Total.values
allAdmsInt = keyAdmInt
Adms2021Int = allAdmsInt[keyDates < np.datetime64('2021-03-01')]
max2021AdmsInts = np.max(rnMean(keyAdmInt,meanWidth))

# keyAdmInt
# keyDates

# ax1.plot(allDatesAdmShift,allAdms,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesAdmShift,meanWidth),rnMean(allAdms,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDatesAdm,100*allAdms/max2021Adms,'r.:',markersize=4,linewidth=0.5,label='Indlæggelser')
ax1.plot(rnTime(keyDates,meanWidth),100*rnMean(allAdmsInt,meanWidth)/max2021AdmsInts,'xkcd:violet',label=f'ICU admissions, {meanWidth} day running mean')


# allAdms = keyAdmInt.Total.values
allDeas2 = allDeas
Dea2021Int = allDeas2[allDatesDea < np.datetime64('2021-03-01')]
max2021DeaInts = np.max(rnMean(allDeas,meanWidth))

# ax1.plot(rnTime(allDatesDeaShift[:-1],meanWidth),rnMean(allDeas[:-1],meanWidth),'k',label=f'2020/2021, {meanWidth} dages gennemsnit')
ax1.plot(rnTime(allDatesDea,meanWidth),100*rnMean(allDeas2,meanWidth)/max2021DeaInts,'r',label=f'Deaths, {meanWidth} day running mean')


# ax1.plot(allDatesShift,allCases,'k.:',markersize=4,linewidth=0.5,label='2020/2021')
# ax1.plot(rnTime(allDatesShift,meanWidth),rnMean(allCases,meanWidth),'k',label=f'2020/2021, {meanWidth} day running mean')
# ax1.plot(allDates,allCases,'b.:',markersize=4,linewidth=0.5,label='2021/2022')
# ax1.plot(rnTime(allDates,meanWidth),rnMean(allCases,meanWidth),'b',label=f'2021/2022, {meanWidth} day running mean')

ax1.legend(loc='upper left')
ax1.grid()
ax1.set_ylabel('% of maximal value in winter 2020/2021')
ax1.set_ylim(bottom=0)
firstDate = np.datetime64('2020-11-01')
curXticks = np.arange(np.datetime64('2020-11'),np.datetime64('2022-11'))
ax1.set_xticks(curXticks)
ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.spines['top'].set_visible(False) 
ax1.spines['right'].set_visible(False)


if saveFigures:
    plt.savefig(path_figs+'CasesHospCompare_English2')

# maxAvg = np.max(rnMean(allCases,meanWidth))
# ax1.set_ylim(top=maxAvg*1.1)

firstDate = np.datetime64('2021-10-01')
ax1.set_xlim([firstDate,lastDate])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
if saveFigures:
    plt.savefig(path_figs+'CasesHospCompare_English2_zoom')

# %%
# # allDeas2[allDates < np.datetime64('2021-03-01')]
# plt.figure()
# plt.plot(allDates,allDeas2)


print('Done making figures for winter comparison')