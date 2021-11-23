# %%
# Load packages and settings
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)
import seaborn as sns


import matplotlib.pyplot as plt
# %matplotlib widget
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
# %matplotlib widget
from matplotlib import cm # Colormaps
import matplotlib.colors as colors
# cmap = plt.cm.get_cmap('Dark2',len(ageGroups))

import locale
import matplotlib.dates as mdates
locale.setlocale(locale.LC_TIME,"Danish")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

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
# dfAdm = pd.read_excel('Admitted\Admitted.xlsx')
# dfAdm = dfAdm.transpose()
# dfAdm.columns = dfAdm.iloc[0]
# dfAdm = dfAdm.drop(['Aldersgruppe']) 
# curDates =  pd.to_datetime(dfAdm.index,format='%d_%m_%Y')

# %%
# # Collect the cases, tests and positive percentage from all the "cases by age" files

# # Load most recent
# latestsubdir = list(os.walk(path_data))[0][1][-1]
# df_recentCasesByAge = pd.read_csv(path_data+latestsubdir+'/Cases_by_age.csv',delimiter = ';',dtype=str)


# ageGroups = df_recentCasesByAge.transpose().iloc[0]

# dfCase = pd.DataFrame(columns=ageGroups.values)
# dfTest = pd.DataFrame(columns=ageGroups.values)
# dfPosP = pd.DataFrame(columns=ageGroups.values)

# for subdir, dirs, files in os.walk(path_data):
#     if not len(files) == 0:
#         latestdir = subdir
#         latestDate = pd.to_datetime(subdir[-10:])
#         curdf = pd.read_csv(latestdir+'/Cases_by_age.csv',delimiter = ';',dtype=str)
        
#         curdf['Antal_bekræftede_COVID-19'] = pd.to_numeric(curdf['Antal_bekræftede_COVID-19'].astype(str).apply(lambda x: x.replace('.','')))
#         curdf['Antal_testede'] = pd.to_numeric(curdf['Antal_testede'].astype(str).apply(lambda x: x.replace('.','')))
#         curdf['Procent_positive'] = pd.to_numeric(curdf['Procent_positive'].astype(str).apply(lambda x: x.replace(',','.')))
        
#         dfCase.loc[latestDate] = curdf['Antal_bekræftede_COVID-19'].values
#         dfTest.loc[latestDate] = curdf['Antal_testede'].values
#         dfPosP.loc[latestDate] = curdf['Procent_positive'].values

# allDates = dfCase.index

# dfCaseDiff = dfCase.diff().iloc[1:]
# dfTestDiff = dfTest.diff().iloc[1:]
# dfPosPDiff = dfPosP.diff().iloc[1:]
# plotDatesCase = allDates[1:]

# # plt.figure()
# # meanWidth = 7
# # plt.plot(plotDatesCase,dfCaseDiff['I alt'],'k.')
# # plt.plot(rnTime(plotDatesCase,meanWidth),rnMean(dfCaseDiff['I alt'],meanWidth),'k')

# # plt.ylim(bottom=0)
# # plt.xlim(left=np.datetime64('2021-07-01'))
# # plt.show()

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

meanWidth = 7

fig,ax1 = plt.subplots()
ax1.plot(dfCase.Date,dfCase.NewPositive,'k.')
ax1.plot(rnTime(dfCase.Date,meanWidth),rnMean(dfCase.NewPositive,meanWidth),'k')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

ax1.set_ylim(bottom=0)
ax1.set_xlim(left=np.datetime64('2021-07-01'))

plt.tight_layout() 

# %%

fig,ax1 = plt.subplots(1,1)

meanWidth = 7

allDates = dfCase.Date 
# prevYearIndex = (allDates < np.datetime64('2021-01-01'))
prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
prevYearDates = allDates[prevYearIndex].values
prevYearSum = dfCase[prevYearIndex]['NewPositive'].values

# thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
thisYearDates = allDates[thisYearIndex].values
thisYearSum = dfCase[thisYearIndex]['NewPositive'].values

ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.:',markersize=4,linewidth=0.5,label='2020')
ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates,thisYearSum,'b.',label='2021')
# ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.:',markersize=4,linewidth=0.5,label='2021')
ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

ax1.legend()
ax1.grid()
ax1.set_ylabel('Antal tilfælde')
ax1.set_ylim(bottom=0)
ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.tight_layout()
plt.savefig(path_figs+'Cases_2020and2021')


# ax1.set_ylim(top=2500)
ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Cases_2020and2021_zoom')

# ax1.set_ylim(top=4500)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Cases_2020and2021_zoom2')

# %%

fig,ax1 = plt.subplots(1,1)

meanWidth = 7

allDates = dfCase.Date 
# prevYearIndex = (allDates < np.datetime64('2021-01-01'))
prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
prevYearDates = allDates[prevYearIndex].values
prevYearSum = dfCase[prevYearIndex]['PosPct'].values

# thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
thisYearDates = allDates[thisYearIndex].values
thisYearSum = dfCase[thisYearIndex]['PosPct'].values

ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.:',markersize=4,linewidth=0.5,label='2020')
ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates,thisYearSum,'b.',label='2021')
# ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.:',markersize=4,linewidth=0.5,label='2021')
ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

ax1.legend()
ax1.grid()
ax1.set_ylabel('Positive procent (PCR-test) [%]')
ax1.set_ylim(bottom=0)
ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.tight_layout()
plt.savefig(path_figs+'PosPct_2020and2021')


ax1.set_ylim(top=5)
ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'PosPct_2020and2021_zoom')

ax1.set_ylim(top=4.5)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'PosPct_2020and2021_zoom2')

# %%

fig,ax1 = plt.subplots(1,1)

meanWidth = 7

allDates = dfCase.Date 
# prevYearIndex = (allDates < np.datetime64('2021-01-01'))
prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
prevYearDates = allDates[prevYearIndex].values
prevYearSum = dfCase[prevYearIndex]['Tested'].values

# thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
thisYearDates = allDates[thisYearIndex].values
thisYearSum = dfCase[thisYearIndex]['Tested'].values

ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.:',markersize=4,linewidth=0.5,label='2020')
ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates,thisYearSum,'b.',label='2021')
# ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.:',markersize=4,linewidth=0.5,label='2021')
ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

ax1.legend()
ax1.grid()
ax1.set_ylabel('Antal PCR-testede')
ax1.set_ylim(bottom=0)
ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.tight_layout()
plt.savefig(path_figs+'Testede_2020and2021')


ax1.set_ylim(top=200000)
ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Testede_2020and2021_zoom')

ax1.set_ylim(top=200000)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Testede_2020and2021_zoom2')

# %%

fig,ax1 = plt.subplots(1,1)

meanWidth = 7

allDates = dfAdm.Dato
# prevYearIndex = (allDates < np.datetime64('2021-01-01'))
prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
prevYearDates = allDates[prevYearIndex].values
prevYearSum = dfAdm[prevYearIndex]['Total'].values

# thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
thisYearDates = allDates[thisYearIndex].values
thisYearSum = dfAdm[thisYearIndex]['Total'].values

ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.:',markersize=4,linewidth=0.5,label='2020')
ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates,thisYearSum,'b.',label='2021')
# ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.:',markersize=4,linewidth=0.5,label='2021')
ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

ax1.legend()
ax1.grid()
ax1.set_ylabel('Antal nyindlæggelser')
ax1.set_ylim(bottom=0)
ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.tight_layout()
plt.savefig(path_figs+'Admissions_2020and2021')


ax1.set_ylim(top=75)
ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Admissions_2020and2021_zoom')

ax1.set_ylim(top=200)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Admissions_2020and2021_zoom2')

# %%

fig,ax1 = plt.subplots(1,1)

meanWidth = 14

allDates = dfDea.Dato
# prevYearIndex = (allDates < np.datetime64('2021-01-01'))
prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
prevYearDates = allDates[prevYearIndex].values
prevYearSum = dfDea[prevYearIndex]['Antal_døde'].values

# thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
thisYearDates = allDates[thisYearIndex].values
thisYearSum = dfDea[thisYearIndex]['Antal_døde'].values

ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.:',markersize=4,linewidth=0.5,label='2020')
ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates,thisYearSum,'b.',markersize=3,label='2021')
# ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.:',markersize=4,linewidth=0.5,label='2021')
ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

ax1.legend()
ax1.grid()
ax1.set_ylabel('Antal dødsfald')
ax1.set_ylim(bottom=0)
ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.tight_layout()
plt.savefig(path_figs+'Deaths_2020and2021')


ax1.set_ylim(top=10)
ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Deaths_2020and2021_zoom')

ax1.set_ylim(top=20)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
plt.tight_layout()
plt.savefig(path_figs+'Deaths_2020and2021_zoom2')

# %% [markdown]
# # Hospitalbelægning

# %%
# Collect the cases, tests and positive percentage from all the "02_hospitalsbelaegning" files

# Load most recent
latestsubdir = list(os.walk(path_dash))[0][1][-1]

latestsubdir
df_recentHosps = pd.read_csv(path_dash+latestsubdir+'\\Regionalt_DB\\02_hospitalsbelaegning.csv',delimiter = ';',dtype=str,encoding='latin1')
df_recentHosps

# ageGroups = df_recentCasesByAge.transpose().iloc[0]

# dfCase = pd.DataFrame(columns=ageGroups.values)
# dfTest = pd.DataFrame(columns=ageGroups.values)
# dfPosP = pd.DataFrame(columns=ageGroups.values)

allDates = []
allHosps = []

for subdir, dirs, files in os.walk(path_dash):
    if not len(files) == 0:
        latestdir = subdir
        if (latestdir[-12:] == 'Regionalt_DB'): # To ensure it's only once per date
            thisdir = latestdir  
            latestDate = pd.to_datetime(latestdir[-23:-13])
            curdf =  pd.read_csv(latestdir+'\\02_hospitalsbelaegning.csv',delimiter = ';',dtype=str,encoding='latin1')
            
            curSum = curdf.Indlagte.astype(int).sum()
            
            allDates.append(latestDate)
            allHosps.append(curSum)


# %%
fig,ax1 = plt.subplots()

ax1.plot(allDates,allHosps,'k.-')

ax1.set_ylim(bottom=0)
ax1.set_ylabel('Hospitalsbelægning (Antal indlagte)')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))

ax1.grid()
plt.tight_layout()

plt.savefig(path_figs+'Hospitalbelaegning')

ax1.set_ylim(top=350)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
plt.savefig(path_figs+'Hospitalbelaegning_zoom')

ax1.set_ylim(top=1000)
ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
plt.savefig(path_figs+'Hospitalbelaegning_zoom2')



# %% [markdown]
# # Nyindlæggelser

# %%

firstDate = dfAdm.Dato.min() 
lastDate =  dfAdm.Dato.max() - np.timedelta64(1,'D')

# firstDate = np.datetime64('2020-11-01') 
# lastDate =  np.datetime64('2021-02-01')

# firstDate = np.datetime64('2021-05-01') 
# lastDate =  np.datetime64('2021-11-01')

dfAdm_slice = dfAdm[(dfAdm.Dato >= firstDate) & (dfAdm.Dato <= lastDate)]
# print(len(dfAdm_slice))
dfCase_slice = dfCase[(dfCase.Date >= firstDate) & (dfCase.Date <= lastDate)]
# print(len(dfCase_slice))
# dfDea_slice = dfDea[(dfDea.Dato >= firstDate) & (dfDea.Dato <= lastDate)]
# print(len(dfDea_slice))

curDates =dfAdm_slice.Dato

# fig,ax1 = plt.subplots(1,1)

# for curDelay in np.arange(1,29,7):
#     # ax1.plot(dfCase_slice.NewPositive[:-curDelay],dfAdm_slice.Total[curDelay:],'*',label=curDelay)
#     ax1.plot(rnTime(dfCase_slice.NewPositive[:-curDelay],7),rnMean(dfAdm_slice.Total[curDelay:],7),label=curDelay) 
    
# ax1.legend()

curDelay = 7
curDates =dfAdm_slice.Dato[:-curDelay]
curCase = dfCase_slice.NewPositive[:-curDelay]
curAdm = dfAdm_slice.Total[curDelay:]

curCaseNorm = curCase/curCase.max()
curAdmNorm = curAdm/curAdm.max()

meanWidth = 7

fig,ax1 = plt.subplots(1,1,figsize=(20,10))
ax1.plot(curDates,curAdmNorm,'.r',markersize=4)
ax1.plot(rnTime(curDates,meanWidth),rnMean(curAdmNorm,meanWidth),'r',label='Indlæggelser')
ax1.plot(curDates,curCaseNorm,'.b',markersize=4)
ax1.plot(rnTime(curDates,meanWidth),rnMean(curCaseNorm,meanWidth),'b',label='Tilfælde')
# ax1.plot(curDates,curAdm,'.r')
# ax1.plot(curDates,curCase,'.b')
ax1.set_ylim(bottom=0)
ax1.legend()

curXticks = np.arange(np.datetime64('2020-02'),np.datetime64('2022-01'))
ax1.set_xticks(curXticks)

ax1.set_ylabel('Normaliseret antal')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))




# fig,(ax1,ax2) = plt.subplots(2,1)
# ax1.plot(curDates,curAdmNorm,'.r')
# ax1.plot(rnTime(curDates,meanWidth),rnMean(curAdmNorm,meanWidth),'r')
# ax1.plot(curDates,curCaseNorm,'.b')
# ax1.plot(rnTime(curDates,meanWidth),rnMean(curCaseNorm,meanWidth),'b')
# # ax1.plot(curDates,curAdm,'.r')
# # ax1.plot(curDates,curCase,'.b')

# ax2.plot(curCase,curAdm,'*')
# ax2.plot(rnTime(curCase,7),rnMean(curAdm,7))
# # ax1.plot(dfCase_slice.NewPositive,dfAdm_slice.Total,'*-')

# ax2.set_xlabel('Tilfælde')
# ax2.set_ylabel('Nyindlæggelser')

# ax2.set_xlim(left=0)
# ax2.set_ylim(bottom=0)

# fig,ax1 = plt.subplots(1,1)

# meanWidth = 7

# rmTime = rnTime(dfAdm['Dato'],meanWidth)
# rmAdm = rnMean(dfAdm['Total'].values,meanWidth)

# ax1.plot(rmTime,rmAdm)

plt.savefig(path_figs+'CasesOgIndlagte')

# %%

# firstDate = dfAdm.Dato.min() 
# lastDate =  dfAdm.Dato.max() - np.timedelta64(1,'D')

firstDate = np.datetime64('2020-04-01') 
lastDate =  np.datetime64('2021-07-01')

# firstDate = np.datetime64('2021-02-01') 
# lastDate =  np.datetime64('2021-11-01')

firstDate = np.datetime64('2020-04-01') 
lastDate =  np.datetime64('2021-11-01')

dfAdm_slice = dfAdm[(dfAdm.Dato >= firstDate) & (dfAdm.Dato <= lastDate)]
# print(len(dfAdm_slice))
dfCase_slice = dfCase[(dfCase.Date >= firstDate) & (dfCase.Date <= lastDate)]
# print(len(dfCase_slice))
dfDea_slice = dfDea[(dfDea.Dato >= firstDate) & (dfDea.Dato <= lastDate)]
# print(len(dfDea_slice))

curDates =dfAdm_slice.Dato

# fig,ax1 = plt.subplots(1,1)

# for curDelay in np.arange(1,29,7):
#     # ax1.plot(dfCase_slice.NewPositive[:-curDelay],dfAdm_slice.Total[curDelay:],'*',label=curDelay)
#     ax1.plot(rnTime(dfCase_slice.NewPositive[:-curDelay],7),rnMean(dfAdm_slice.Total[curDelay:],7),label=curDelay) 
    
# ax1.legend()

curDelay = 10
# curDelay = 21
curDates =dfAdm_slice.Dato[:-curDelay]
curAdm = dfAdm_slice.Total[:-curDelay]
curDea = dfDea_slice.Antal_døde[curDelay:]

curDeaNorm = curDea/curDea.max()
curAdmNorm = curAdm/curAdm.max()

meanWidth = 7

fig,ax1 = plt.subplots(1,1,figsize=(20,10))
ax1.plot(curDates,curAdmNorm,'.r')
ax1.plot(rnTime(curDates,meanWidth),rnMean(curAdmNorm,meanWidth),'r',label='Indlæggelser')
ax1.plot(curDates,curDeaNorm,'.k')
ax1.plot(rnTime(curDates,meanWidth),rnMean(curDeaNorm,meanWidth),'k',label='Dødsfald')
# ax1.plot(curDates,curAdm,'.r')
# ax1.plot(curDates,curCase,'.b')
ax1.legend()

ax1.set_ylim(bottom=0)

curXticks = np.arange(np.datetime64('2020-02'),np.datetime64('2022-01'))
ax1.set_xticks(curXticks)

ax1.set_ylabel('Normaliseret antal')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# ax2.plot(curCase,curAdm,'*')
# ax2.plot(rnTime(curCase,7),rnMean(curAdm,7))
# # ax1.plot(dfCase_slice.NewPositive,dfAdm_slice.Total,'*-')

# ax2.set_xlabel('Tilfælde')
# ax2.set_ylabel('Nyindlæggelser')

# ax2.set_xlim(left=0)
# ax2.set_ylim(bottom=0)

# fig,ax1 = plt.subplots(1,1)

# meanWidth = 7

# rmTime = rnTime(dfAdm['Dato'],meanWidth)
# rmAdm = rnMean(dfAdm['Total'].values,meanWidth)

# ax1.plot(rmTime,rmAdm)
plt.savefig(path_figs+'IndlagteOgDoedsfald')


# %%

# allDates = dfAdm.Dato
# # prevYearIndex = (allDates < np.datetime64('2021-01-01'))
# prevYearIndex = (allDates < (np.datetime64('2021-01-01') + np.timedelta64(meanWidth,'D')))
# prevYearDates = allDates[prevYearIndex].values
# prevYearSum = dfAdm[prevYearIndex]['Total'].values

# # thisYearIndex = (allDates >= np.datetime64('2021-01-01'))
# thisYearIndex = (allDates >=  (np.datetime64('2021-01-01') - np.timedelta64(meanWidth,'D')))
# thisYearDates = allDates[thisYearIndex].values
# thisYearSum = dfAdm[thisYearIndex]['Total'].values

# ax1.plot(prevYearDates+np.timedelta64(365,'D'),prevYearSum,'k.',label='2020')
# ax1.plot(rnTime(prevYearDates+np.timedelta64(365,'D'),meanWidth),rnMean(prevYearSum,meanWidth),'k',label=f'2020, {meanWidth} dages gennemsnit')
# # ax1.plot(thisYearDates,thisYearSum,'b.',label='2021')
# # ax1.plot(rnTime(thisYearDates,meanWidth),rnMean(thisYearSum,meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')
# ax1.plot(thisYearDates[:-1],thisYearSum[:-1],'b.',label='2021')
# ax1.plot(rnTime(thisYearDates[:-1],meanWidth),rnMean(thisYearSum[:-1],meanWidth),'b',label=f'2021, {meanWidth} dages gennemsnit')

# ax1.legend()
# ax1.grid()
# ax1.set_ylabel('Antal nyindlæggelser')
# # ax1.set_ylim(bottom=0)
# ax1.set_xlim([np.datetime64('2021-01-01'),np.datetime64('2022-01-01')])
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))

# ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
# plt.yscale('log')
# # plt.tight_layout()
# # plt.savefig(path_figs+'Admissions_2020and2021')


# # ax1.set_ylim(top=75)
# # ax1.set_xlim([np.datetime64('2021-06-01'),np.datetime64('2022-01-01')])
# # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
# # plt.tight_layout()
# # plt.savefig(path_figs+'Admissions_2020and2021_zoom')

# # ax1.set_ylim(top=120)
# # ax1.set_xlim([np.datetime64('2021-09-01'),np.datetime64('2022-01-01')])
# # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%B'))
# # plt.tight_layout()
# # plt.savefig(path_figs+'Admissions_2020and2021_zoom2')

# %% [markdown]
# # Age-splits

# %%

latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
latestdir

dfBreakCase = pd.read_csv(latestdir+'\\Gennembruds_DB\\04_bekræftede_tilfælde_pr_vaccinationsstatus_pr_aldersgrp_pr_uge.csv',delimiter = ';',encoding='latin1')
dfBreakCase.tail()
# dfCase = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
# dfCase = dfCase.iloc[:-2]
# dfCase['NewPositive'] = pd.to_numeric(dfCase['NewPositive'].astype(str).apply(lambda x: x.replace('.','')))
# dfCase['Tested'] = pd.to_numeric(dfCase['Tested'].astype(str).apply(lambda x: x.replace('.','')))
# dfCase['PosPct'] = pd.to_numeric(dfCase['PosPct'].astype(str).apply(lambda x: x.replace(',','.')))
# dfCase['Date'] =  pd.to_datetime(dfCase.Date,format='%Y-%m-%d')
# testDates = dfCase['Date']


# %%
curYears = dfBreakCase.Uge.apply(lambda x: int(x[:4]))

# Remove everything before 2021
curdf = dfBreakCase[curYears > 2020].copy()


weekNums = curdf.Uge.apply(lambda x: int(x[-2:]))

    


# Uge 36: 07-09-2021 (tirsdag) og 12-09-2021 (søndag)

weekOffset = weekNums - 36
allDates = weekOffset.apply(lambda x:np.datetime64('2021-09-07') + np.timedelta64(x*7,'D'))
allDates = weekOffset.apply(lambda x:np.datetime64('2021-09-12') + np.timedelta64(x*7,'D'))

allDates
curdf['Dato'] = allDates

# %%
dfVacc = curdf[curdf.Vaccinationsstatus == 'Forventet fuld effekt']
dfUvacc = curdf[curdf.Vaccinationsstatus == 'Ikke vaccineret']
# curdf.Vaccinationsstatus.unique()

fig,ax1 = plt.subplots()

for k in dfVacc.groupby('Aldersgruppe'):
    thisage = k[0]
    thisdf = k[1]
    
    thisDays = thisdf.Dato
    thisCase = thisdf['Bekræftede tilfælde']
    
    ax1.plot(thisDays,thisCase,'.-',label=thisage+', Vacc')
    
for k in dfUvacc.groupby('Aldersgruppe'):
    thisage = k[0]
    thisdf = k[1]
    
    thisDays = thisdf.Dato
    thisCase = thisdf['Bekræftede tilfælde']
    
    ax1.plot(thisDays,thisCase,'.-',label=thisage+', Uvacc')
    
ax1.legend()
fig.tight_layout()
    
    
    

# %%
# dfVacc[dfVacc.Aldersgruppe == '12-15']

print(dfCase.tail(5))
print(dfAdm.tail(5))

print('Script done')
