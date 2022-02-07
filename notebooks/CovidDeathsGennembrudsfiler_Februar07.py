# %%
%matplotlib widget
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
# import datetime

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
# rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 
# rootdir_data = "D:\Pandemix\Github\DanskeData\\"
rootdir_data = "C:\\Users\\rakrpe\\OneDrive - Roskilde Universitet\\Documents\\PandemiX\\GithubRepos\\PandemiX\\DanskeData\\"

path_data = rootdir_data + "ssi_data\\"
path_dash = rootdir_data + "ssi_dashboard\\"
path_vacc = rootdir_data + "ssi_vacc\\"
path_figs = os.getcwd() +"\\..\\Figures\\" 

# %%
import datetime

# Data is (only) in the file from the most recent tuesday. 
# Should be made smarter, but here hardcoded
# tuePath = 'SSI_data_2022-01-25'
# Now automatic finding of latest tuesday:
for k in range(0,7):
    dayToCheck = np.datetime64('today')-np.timedelta64(k,'D')
    thisWeekDay = (dayToCheck).astype(datetime.datetime).isoweekday()    
    if (thisWeekDay == 2):
        tuePath = 'SSI_data_'+str(dayToCheck)

print(f'Path to latest Tuesday was {tuePath}')

dirPath = path_data + tuePath + '\\'

df1 = pd.read_csv(dirPath+'gennembrudsinfektioner_table1.csv',delimiter=';')

df2_C = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_cases.csv',delimiter=';')
df2_H = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_indlagte.csv',delimiter=';')
df2_D = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_dode.csv',delimiter=';')
df2_R = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_repositive.csv',delimiter=';')
df2_Int = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_intensiv.csv',delimiter=';')
df2_T = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_antal_tests.csv',delimiter=';')

df3 = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_alle.csv',delimiter=';',decimal=",")
df3_C = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_cases.csv',delimiter=';',decimal=",")
df3_H = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_indlagte.csv',delimiter=';',decimal=",")
df3_D = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_dode.csv',delimiter=';',decimal=",")
df3_Int = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_intensiv.csv',delimiter=';',decimal=",")
df3_T = pd.read_csv(dirPath+'gennembrudsinfektioner_table2_incidence_tests.csv',delimiter=';',decimal=",")

# %%
# Population counts, from Danmark Statistik. For 2021
popdf1 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del1.csv',header=None,encoding='latin1',delimiter=';')
popdf2 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del2.csv',header=None,encoding='latin1',delimiter=';')

popdf = pd.concat([popdf1,popdf2])

popdf = popdf.rename(columns={0:"Kommune",1:'Alder',2:'Antal'})
popdf['AlderKort'] = popdf.Alder.apply(lambda x: int(str(x).split(' ')[0]))
totCounts = popdf.groupby('Kommune').sum()

# Also collect national numbers
popdf_nat = popdf.groupby('Alder').sum()
popdf_nat['AlderKort'] =[int(str(x).split(' ')[0]) for x in popdf_nat.index]
popdf_nat = popdf_nat.sort_values('AlderKort')

def getPopSizeNational(minAlder=0,maxAlder=125):
    return popdf_nat[(popdf_nat.AlderKort >= minAlder) & (popdf_nat.AlderKort <= maxAlder)].Antal.sum()


# %%
df_dkstat = pd.read_csv('DK_Stat_Deaths.csv',encoding='latin1',delimiter=';')
curYearWeek = df_dkstat.iloc[:-3:3,0]
df_dkstat = df_dkstat.iloc[2:-1:3,3:]

df_temp = pd.DataFrame()
df_temp['Week'] = curYearWeek.values
curYears = df_temp['Week'].apply(lambda x: x[:4]).astype(int)
curWeeks = df_temp['Week'].apply(lambda x: x[5:]).astype(int)
dkstatDates = pd.to_datetime((curYears*100+curWeeks).astype(str)+'1',format='%G%V%u')
df_dkstat = df_dkstat.transpose()
df_dkstat.columns =list(dkstatDates)
df_dkstat = df_dkstat.iloc[1:]
# Function for getting all cause mortality in a specific range
def getAllCause(minAge=0,maxAge=125):

    firstAges = np.array([int(x.split('-')[0]) for x in df_dkstat.index[:-1]])
    lastAges = np.array([int(x.split('-')[1].split(' ')[0]) for x in df_dkstat.index[:-1]])

    firstIndex = np.where(firstAges == minAge)[0][0]
    if (lastAges == maxAge).any():
        lastIndex = np.where(lastAges == maxAge)[0][0]
    else:
        lastIndex = len(df_dkstat)

    curSum = df_dkstat.iloc[firstIndex:lastIndex].sum()
    curDates = curSum.index
    curCount = curSum.values

    return curCount,curDates

getAllCause(0,5)
# getAllCause(4,20)
df_dkstat.iloc[0]
df_dkstat

# %%
# Get all cause from MOMO
df_momo = pd.read_csv('MOMOdata_ny.csv',delimiter=';')
df_momo['nbc'] = pd.to_numeric(df_momo['nbc'].str.replace(',','.'))
df_momo['Pnb'] = pd.to_numeric(df_momo['Pnb'].str.replace(',','.'))
df_momo['YearWeek'] = df_momo.YoDi.astype(str) + df_momo.WoDi.apply(lambda x: '{0:0>2}'.format(x))
df_momo['Date'] = pd.to_datetime(df_momo.YearWeek.astype(str)+'1',format='%G%V%u')
print(df_momo.columns)
allAges = df_momo.group.unique()
print(allAges)

# %%
# # Make a test plot of momo data
# fig,ax1 = plt.subplots()
# # groupbyOb = list(df_momo.groupby('group'))
# # curAge = groupbyOb[13][0]
# # curdf =  groupbyOb[13][1]
# for curAge,curdf in df_momo.groupby('group'):
#     if curAge == 'Total':
#         continue
#     ax1.plot(curdf.Date,curdf.nb,label=curAge)
#     # ax1.plot(curdf.Date,curdf.nbc,':')
#     # ax1.set_title(curAge)
# ax1.legend()

# %%

def getAllCauseMOMO(minAge=0,maxAge=125):
    firstAge = np.array([int(x.split('to')[0]) for x in allAges[:-2]])
    lastAge = np.array([int(x.split('to')[1]) for x in allAges[:-2]])

    firstIndex = 0
    lastIndex = -1

    if (minAge == 95):
        firstIndex = -2
    elif (firstAge == minAge).any():
        firstIndex = np.where(firstAge == minAge)[0][0]
    else:
        print('Incorrect minimum age, using 0')
        
    if (maxAge == 125):
        lastIndex = -1
    elif (lastAge == maxAge).any():
        lastIndex = np.where(lastAge == maxAge)[0][0]+1
    else:
        print('Incorrect maximum age, using maximum')

        # toReturnCount = df_momo[df_momo.group == '95P'].nbc
        # toReturnDates = df_momo[df_momo.group == '95P'].Date
    curdf = df_momo[df_momo.group.isin(allAges[firstIndex:lastIndex])].groupby('Date').sum()

    toReturnDates = curdf.index
    toReturnCount_raw = curdf.nb
    toReturnCount = curdf.nbc

    return toReturnCount.values,toReturnDates

# %%
# Since order was wrong in the beginning of 2022, we first need the correct order...
# weekNames = df.År.astype(str)+'-W'+df.Uge.apply(lambda x: f"{int(x):02d}")

weekDTs = [np.datetime64(datetime.datetime.strptime(d[-4:] + '-W'+d[4:6]+'-1', "%Y-W%W-%w")) for d in df1.Ugenummer]

curOrder = np.argsort(weekDTs)
        
sNone = 'Ingen vaccination'        
sOne = 'Første vaccination'  
sTwo = 'Anden vaccination'
sFull = 'Fuld effekt efter primært forløb'   
sReva = 'Fuld effekt efter revaccination'


ageGroups = df2_C.Aldersgruppe.values
# print(ageGroups)
# weekNames = df1.Ugenummer
weekNames = df1.Ugenummer.values[curOrder]
weekNamesShort = [x[4:6] for x in weekNames]
wInt = [int(x[4:6]) for x in weekNames]
wIntRange = np.arange(len(wInt))

allDates = np.array(weekDTs)[curOrder]
print(weekNames)

# Make function for gettings particular parts
def getTimeSeries(thisdf=df2_C,curStatus='Ingen vaccination',curAge='Alle',weekNames=weekNames):
    
    agedf = thisdf[thisdf.Aldersgruppe==curAge]
    allVals = []
    for curWeek in weekNames:
        toAdd = agedf[curWeek+'_'+curStatus].values[0]
        allVals.append(toAdd)
    allVals = np.array(allVals)

    return allVals

def getTimeSeriesAll(thisdf=df2_C,curAge='Alle',weekNames=weekNames):
    return getTimeSeries(thisdf,sNone,curAge,weekNames)+getTimeSeries(thisdf,sOne,curAge,weekNames)+getTimeSeries(thisdf,sTwo,curAge,weekNames)
    

# %%
ssiAges = df2_C.Aldersgruppe.unique()
# print(ssiAges)
df_case = pd.DataFrame()
df_case['Date'] = allDates
df_case['0-19'] = getTimeSeriesAll(df2_C,'0-5') + getTimeSeriesAll(df2_C,'6-11') + getTimeSeriesAll(df2_C,'12-15') + getTimeSeriesAll(df2_C,'16-19') 
df_case['20-39'] = getTimeSeriesAll(df2_C,'20-29') + getTimeSeriesAll(df2_C,'30-39')
df_case['40-59'] = getTimeSeriesAll(df2_C,'40-49') + getTimeSeriesAll(df2_C,'50-59')
df_case['60-69'] = getTimeSeriesAll(df2_C,'60-64') + getTimeSeriesAll(df2_C,'65-69')
# df_case['70-125'] = getTimeSeriesAll(df2_C,'70-79') + getTimeSeriesAll(df2_C,'80+')
df_case['70-79'] = getTimeSeriesAll(df2_C,'70-79')
df_case['80-125'] = getTimeSeriesAll(df2_C,'80+')

df_death = pd.DataFrame()
df_death['Date'] = allDates
df_death['0-19'] = getTimeSeriesAll(df2_D,'0-5') + getTimeSeriesAll(df2_D,'6-11') + getTimeSeriesAll(df2_D,'12-15') + getTimeSeriesAll(df2_D,'16-19') 
df_death['20-39'] = getTimeSeriesAll(df2_D,'20-29') + getTimeSeriesAll(df2_D,'30-39')
df_death['40-59'] = getTimeSeriesAll(df2_D,'40-49') + getTimeSeriesAll(df2_D,'50-59')
df_death['60-69'] = getTimeSeriesAll(df2_D,'60-64') + getTimeSeriesAll(df2_D,'65-69')
# df_death['70-125'] = getTimeSeriesAll(df2_D,'70-79') + getTimeSeriesAll(df2_D,'80+')
df_death['70-79'] = getTimeSeriesAll(df2_D,'70-79') 
df_death['80-125'] = getTimeSeriesAll(df2_D,'80+')

# %%
# Calculate cases relative to population
df_case_rela = df_case.copy()
df_case_rela['0-19'] = df_case['0-19']/getPopSizeNational(0,19)
df_case_rela['20-39'] = df_case['20-39']/getPopSizeNational(20,39)
df_case_rela['40-59'] = df_case['40-59']/getPopSizeNational(40,59)
df_case_rela['60-69'] = df_case['60-69']/getPopSizeNational(60,69)
# df_case_rela['70-125'] = df_case['70-125']/getPopSizeNational(70,125)
df_case_rela['70-79'] = df_case['70-79']/getPopSizeNational(70,79)
df_case_rela['80-125'] = df_case['80-125']/getPopSizeNational(80,125)

# %%
# Collect all cause in a similar dataframe
df_allcause = pd.DataFrame()
df_allcause['Date'] = getAllCauseMOMO(0,19)[1]
df_allcause['0-19'] = getAllCauseMOMO(0,19)[0]
df_allcause['20-39'] = getAllCauseMOMO(20,39)[0]
df_allcause['40-59'] = getAllCauseMOMO(40,59)[0]
df_allcause['60-69'] = getAllCauseMOMO(60,69)[0]
# df_allcause['70-125'] = getAllCauseMOMO(70,125)[0]
df_allcause['70-79'] = getAllCauseMOMO(70,79)[0]
df_allcause['80-125'] = getAllCauseMOMO(80,125)[0]

# %%
# df_allcause
# getAllCauseMOMO(0,19)[0]

# %%
# allAgesCurrently = df_case.columns[1:]

# # fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(15,15),sharex=True)
# fig,allAxes = plt.subplots(2,2,figsize=(18,10),sharex=True)
# ax1,ax2,ax3,ax4 = allAxes.flatten()
# for curAge in allAgesCurrently:
#     ax1.plot(df_case.Date,df_case[curAge],'.-',label=curAge)
#     ax2.plot(df_case.Date,df_case_rela[curAge],'.-',label=curAge)
#     ax3.plot(df_case.Date,df_death[curAge],'.-',label=curAge)
#     ax4.plot(df_allcause.Date,df_allcause[curAge],'.-',label=curAge)
# ax1.legend(loc = 'upper left')
# ax1.set_ylim(bottom=0)
# ax2.set_ylim(bottom=0)
# ax3.set_ylim(bottom=0)
# ax4.set_ylim(bottom=0)

# ax1.set_ylabel('Cases')
# ax2.set_ylabel('Cases relative to population')
# ax3.set_ylabel('Deaths registrered as covid-deaths')
# ax4.set_ylabel('All-cause mortality (EuroMOMO)')

# ax1.set_xlim(left=df_case.Date[0],right=np.datetime64('today'))
# ax1.grid()
# ax2.grid()
# ax3.grid()
# ax4.grid()

# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

# plt.tight_layout()

# if saveFigures:
#     plt.savefig('Figures\\DataOverview')

# %%
curdf_case = df_case['0-19'].values
print(curdf_case)
np.cumsum(curdf_case[3:])-np.cumsum(curdf_case[:-3])
# from numpy.lib.stride_tricks import sliding_window_view
# curRoll = [x.sum() for x in sliding_window_view(curdf_case, window_shape = 4)]

# %%
def rollSum(curArray,rollWidth = 4):
    curRoll = []
    for x in range(rollWidth-1,len(curArray)):
        curRoll.append(curArray[x+1-rollWidth:x+1].sum())
    curRoll = np.array(curRoll)
    return(curRoll)
rollSum(df_case['0-19'])

# %%
# Calculate 4-week rolling sum of cases relative to population
# df_case_roll = df_case_rela.copy()
df_case_roll = pd.DataFrame()
df_case_roll['Date'] = df_case.Date[3:]
df_case_roll['0-19'] = rollSum(df_case['0-19'])/getPopSizeNational(0,19)
df_case_roll['20-39'] = rollSum(df_case['20-39'])/getPopSizeNational(20,39)
df_case_roll['40-59'] = rollSum(df_case['40-59'])/getPopSizeNational(40,59)
df_case_roll['60-69'] = rollSum(df_case['60-69'])/getPopSizeNational(60,69)
# df_case_roll['70-125'] = rollSum(df_case['70-125'])/getPopSizeNational(70,125)
df_case_roll['70-79'] = rollSum(df_case['70-79'])/getPopSizeNational(70,79)
df_case_roll['80-125'] = rollSum(df_case['80-125'])/getPopSizeNational(80,125)

# %%
# allAgesCurrently = df_case.columns[1:]

# fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(15,15),sharex=True)
# for curAge in allAgesCurrently:
#     ax1.plot(df_case.Date,df_case[curAge],label=curAge)
#     ax2.plot(df_case.Date,df_case_rela[curAge],label=curAge)
#     ax3.plot(df_case_roll.Date,df_case_roll[curAge],label=curAge)
# ax1.legend()
# ax1.set_ylim(bottom=0)
# ax2.set_ylim(bottom=0)
# ax3.set_ylim(bottom=0)
# ax4.set_ylim(bottom=0)

# ax1.set_ylabel('Cases')
# ax2.set_ylabel('Cases relative to population')
# ax3.set_ylabel('Rolling sum of cases (4-weeks)')
# ax4.set_ylabel('All-cause mortality (EuroMOMO)')

# ax1.set_xlim(left=df_case.Date[0])

allAgesCurrently = df_case.columns[1:]

# fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(15,15),sharex=True)
fig,allAxes = plt.subplots(2,2,figsize=(18,10),sharex=True)
ax1,ax2,ax3,ax4 = allAxes.flatten()

# ax1_1 = ax1.twinx()

for curAge in allAgesCurrently:
    # ax1.plot(df_case.Date,df_case[curAge],'.-',label=curAge)
    ax1.plot(df_case.Date,df_case_rela[curAge],'.-',label=curAge)
    ax2.plot(df_case_roll.Date,df_case_roll[curAge],label=curAge)
    ax3.plot(df_case.Date,df_death[curAge],'.-',label=curAge)
    ax4.plot(df_allcause.Date,df_allcause[curAge],'.-',label=curAge)
ax1.legend(loc = 'upper left')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax4.set_ylim(bottom=0)

# ax1.set_ylabel('Cases')
ax1.set_ylabel('Cases relative to population')
ax2.set_ylabel('4-week sum of cases (relative)')
ax3.set_ylabel('Deaths registrered as covid-deaths')
ax4.set_ylabel('All-cause mortality (EuroMOMO)')

ax1.set_xlim(left=df_case.Date[0],right=np.datetime64('today'))
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

plt.tight_layout()

if saveFigures:
    plt.savefig('Figures\\DataOverview')

# %% [markdown]
# # Viggo's metode

# %%
# df_case.columns[1:]
# 1-P

# %%
df_X = pd.DataFrame()
df_Y = pd.DataFrame()
df_D = pd.DataFrame()
df_C = pd.DataFrame()
df_P = pd.DataFrame()
df_PD = pd.DataFrame()

for curAge in df_case.columns[1:]:
    numRoll = len(df_case_roll)
    curDates = df_case['Date'].values[-numRoll:]
    cur_allcause = df_allcause[curAge].values[-numRoll:]

    cur_inci = df_case_roll[curAge].values
    cur_regiDeath = df_death[curAge].values[-numRoll:]

    D = cur_allcause
    C = cur_regiDeath
    P = cur_inci

    X = (D-C)/(1-P)
    Y = (C-(P*D))/(1-P)
    
    df_X[curAge] = X
    df_Y[curAge] = Y
    df_D[curAge] = D
    df_C[curAge] = C
    df_P[curAge] = P
    df_PD[curAge] = P*D

# %%
len(df_Y)
# len(weekNamesShort)
curWeekNames = weekNamesShort[-len(df_Y):]
curWeekNames

df_Y_non = df_Y.copy()
df_Y_non[df_Y_non < 0] = 0
df_Y_non

# %%
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(13,9))
curDates = df_case_roll.Date.values

curWeekNames = weekNamesShort[-len(df_Y):]
curDates = curWeekNames

Y_sum = df_Y.sum(axis=1)
Y_sum_non = df_Y_non.sum(axis=1)
C_sum = df_C.sum(axis=1)
D_sum = df_D.sum(axis=1)
X_sum = df_X.sum(axis=1)
ax1.fill_between(curDates,Y_sum,color='xkcd:purple',label='Dødsfald af COVID-19')
ax1.fill_between(curDates,C_sum,Y_sum,color='xkcd:green',label='Dødsfald med COVID-19')
# ax1.plot(curDates,Y_sum_non,label='Y, no negative')
# ax1.plot(curDates,df_Y.sum(axis=1),label='Y')
# ax1.plot(curDates,df_C.sum(axis=1),label='C')
# # ax1.plot(curDates,C_sum,label='C')
# ax2.fill_between(curDates,100*np.ones(C_sum.shape),label='C')
# ax2.fill_between(curDates,100*(C_sum-Y_sum)/C_sum,label='C')
ax2.plot(curDates,100*(C_sum-Y_sum)/C_sum,color='k',label='Andel af dødsfald med COVID-19')
# ax2.plot(curDates,100*(C_sum-Y_sum_non)/C_sum,label='No negative')
# ax1.plot(curDates,Y_sum,'Y')
# ax1.plot(curDates,Y_sum,'Y')

ax1.legend(loc='upper left')
# ax2.legend()
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax1.set_ylabel('Ugentlige COVID-19 dødsfald')
ax2.set_ylabel('Andel af dødsfald med COVID-19\naf de registrerede COVID-19-dødsfald [%]')
# ax2.set_ylim(top=100)
ax2.set_ylim(top=50)
ax1.set_xlim(left=curDates[0],right=curDates[-1])

ax2.set_xlabel('Uge')
ax1.grid(axis='y')
ax2.grid(axis='y')
ax1.set_axisbelow(True)
plt.tight_layout()

if saveFigures:
    fig.savefig('Figures/CovidDødsfaldSamlet')

# %%
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(13,9))
curDates = df_case_roll.Date.values

curWeekNames = weekNamesShort[-len(df_Y):]
curDates = curWeekNames

Y_sum = df_Y.sum(axis=1)
Y_sum_non = df_Y_non.sum(axis=1)
C_sum = df_C.sum(axis=1)
D_sum = df_D.sum(axis=1)
X_sum = df_X.sum(axis=1)
ax1.fill_between(curDates,Y_sum_non,color='xkcd:purple',label='Dødsfald af COVID-19')
ax1.fill_between(curDates,C_sum,Y_sum_non,color='xkcd:green',label='Dødsfald med COVID-19')
# ax1.plot(curDates,Y_sum_non,label='Y, no negative')
# ax1.plot(curDates,df_Y.sum(axis=1),label='Y')
# ax1.plot(curDates,df_C.sum(axis=1),label='C')
# # ax1.plot(curDates,C_sum,label='C')
# ax2.fill_between(curDates,100*np.ones(C_sum.shape),label='C')
# ax2.fill_between(curDates,100*(C_sum-Y_sum)/C_sum,label='C')
ax2.plot(curDates,100*(C_sum-Y_sum_non)/C_sum,color='k',label='Andel af dødsfald med COVID-19')
# ax2.plot(curDates,100*(C_sum-Y_sum_non)/C_sum,label='No negative')
# ax1.plot(curDates,Y_sum,'Y')
# ax1.plot(curDates,Y_sum,'Y')

ax1.legend(loc='upper left')
# ax2.legend()
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax1.set_ylabel('Ugentlige COVID-19 dødsfald')
ax2.set_ylabel('Andel af dødsfald med COVID-19\naf de registrerede COVID-19-dødsfald [%]')
# ax2.set_ylim(top=100)
ax2.set_ylim(top=50)
ax1.set_xlim(left=curDates[0],right=curDates[-1])

ax2.set_xlabel('Uge')
ax1.grid(axis='y')
ax2.grid(axis='y')
ax1.set_axisbelow(True)
plt.tight_layout()

if saveFigures:
    fig.savefig('Figures/CovidDødsfaldSamlet_UdenNegative')

# %%
# plt.figure()
# plt.plot(curDatesAll,df_case['70-79'])


# %%
# curAge = '70-79'
plt.close('all')

for curAge in df_case.columns[1:]:
    curDates = df_case_roll.Date.values
    curWeekNames = weekNamesShort[-len(df_Y):]
    curDates = curWeekNames
    curDatesAll = weekNamesShort

    # fig,allAxes = plt.subplots(2,2,sharex=True)
    fig,allAxes = plt.subplots(2,2,figsize=(15,9))

    # ax1,ax2 = allAxes.flatten()
    # ax1,ax2,ax3,ax4 = allAxes.flatten()
    # ax4,ax3,ax1,ax2 = allAxes.flatten()
    ax3,ax4,ax1,ax2 = allAxes.flatten()

    ax3.plot(curDatesAll,df_case[curAge],'k.-')
    ax3_2 = ax3.twinx()
    ax3_2.plot(curDatesAll,100*df_case_rela[curAge],'k.-')

    ax4.plot(curDates,100*df_case_roll[curAge],'k.-')

    ax1.plot(curDates,df_D[curAge],'k.-',label='D: All-cause mortality (EuroMOMO)')
    ax1.plot(curDates,df_X[curAge],'m.-',label='X: Dødsfald urelateret til COVID-19')
    ax2.plot(curDates,df_C[curAge],'k.-',label='C: Dødsfald registreret som COVID-19')
    ax2.plot(curDates,df_Y[curAge],'b.-',label='Y: Dødsfald af COVID-19')
    ax2.plot(curDates,df_C[curAge]-df_Y[curAge],'.-',color='xkcd:dark yellow',label='C-Y: Dødsfald med COVID-19')

    ax1.legend(loc='lower left',fontsize=13)
    ax2.legend(loc='upper left',fontsize=13)

    ax1.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    ax3_2.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    
    curLabel = 'Aldersgruppe: '
    if curAge == '80-125':
        curLabel = curLabel + '80 årige og op'
    else:
        curLabel = curLabel + curAge.split('-')[0] + ' til '+ curAge.split('-')[1] + ' årige'
    fig.suptitle(curLabel,fontsize=28)
    # fig.suptitle(f'Aldersgruppe: {curAge}',fontsize=28)

    ax1.grid(axis='y')
    ax2.grid(axis='y')
    ax3.grid(axis='y')
    ax4.grid(axis='y')

    ax1.set_ylabel('Ugentlige dødsfald')
    ax2.set_ylabel('Ugentlige dødsfald')
    ax3.set_ylabel('Antal ugentlige tilfælde')
    ax3_2.set_ylabel('Ugentlige tilfælde\nper 100 borgere [%]')
    ax4.set_ylabel('4-ugers rullende sum af\ntilfælde per 100 borgere [%]')

    ax1.set_xlim(left=curDates[0],right=curDates[-1])
    ax2.set_xlim(left=curDates[0],right=curDates[-1])
    ax3.set_xlim(left=curDatesAll[3],right=curDatesAll[-1])
    ax4.set_xlim(left=curDates[0],right=curDates[-1])

    ax1.set_xlabel('Uge')
    ax2.set_xlabel('Uge')

    plt.tight_layout()

    if saveFigures:
        fig.savefig(f'Figures\\Overblik{curAge}')

# %%
# plt.close('all')
# curAge = '0-19'
# curAge = '20-39'
# curAge = '40-59'
# curAge = '60-69'
# # curAge = '70-125'
# curAge = '70-79'
# curAge = '80-125'

# numRoll = len(df_case_roll)
# curDates = df_case['Date'].values[-numRoll:]
# cur_allcause = df_allcause[curAge].values[-numRoll:]

# cur_inci = df_case_roll[curAge].values
# cur_regiDeath = df_death[curAge].values[-numRoll:]

# D = cur_allcause
# C = cur_regiDeath
# P = cur_inci

# X = (D-C)/(1-P)
# Y = (C-(P*D))/(1-P)

# # Y + X = D
# # Y + P*X = C 

# # Y + X - Y - P*X = D-C
# # (1-P)X = D-C 
# # X = (D-C)/(1-P)

# # Y + (D-C)/(1-P) = D
# # Y = D - (D-C)/(1-P) 
# # Y = [(1-P)D - D + C]/(1-P)
# # Y = [C-P*D]/(1-P)

# # C - Y = P*D

# # fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
# # ax1.plot(curDates,D,label='All cause mortality (D)')
# ax1.errorbar(curDates,D,np.sqrt(D),label='All cause mortality (D)')
# ax1.plot(curDates,X,label='Non-Covid mortality (X)')
# # ax2.plot(curDates,C,label='Registrered Covid mortality (C)')
# ax2.errorbar(curDates,C,np.sqrt(C),label='Registrered Covid mortality (C)')
# # ax2.plot(curDates,P*X,label='P*X')
# # ax2.plot(curDates,P*D,label='P*D')
# ax2.plot(curDates,Y,label='Actual Covid-mortality (Y)')
# # ax2.plot(curDates,C-Y,label='Misregistrations (C-Y)')
# ax2.errorbar(curDates,C-Y,np.sqrt(C-Y),label='Misregistrations (C-Y)')
# # ax3.fill_between(curDates,100*np.ones(C.shape),label='Covid')
# # ax3.fill_between(curDates,100*(C-Y)/C,label='Non-Covid')


# ax1.legend()
# ax2.legend()
# # ax3.legend()

# ax1.grid()
# ax2.grid()

# # ax3.set_ylim([0,50]) 
# ax1.set_ylim(bottom=0)

# # ax3.set_ylabel('Covid-registrered deaths [%]')

# if saveFigures:
#     fig.savefig(f'Figures/ViggoMetode_{curAge}')

# %%
# curAge = '0-19'
# # curAge = '20-39'
# # curAge = '40-59'
# # curAge = '60-69'
# # curAge = '70-125'

# numRoll = len(df_case_roll)
# curDates = df_case['Date'].values[-numRoll:]
# cur_allcause = df_allcause[curAge].values[-numRoll:]

# cur_inci = df_case_roll[curAge].values
# cur_regiDeath = df_death[curAge].values[-numRoll:]

# D = cur_allcause
# C = cur_regiDeath
# P = cur_inci

# X = (D-C)/(1-P)
# Y = (C-(P*D))/(1-P)
# X


# # fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
# ax1.plot(curDates,P,label='P')
# # ax1.plot(curDates,X,label='Non-Covid mortality (X)')
# # ax2.plot(curDates,C,label='Registrered Covid mortality (C)')
# # ax2.plot(curDates,Y,label='Actual Covid-mortality (Y)')
# # ax2.plot(curDates,C-Y,label='Misregistrations (C-Y)')
# # ax3.fill_between(curDates,100*np.ones(C.shape),label='Covid')
# # ax3.fill_between(curDates,100*(C-Y)/C,label='Non-Covid')


# # ax1.legend()
# # ax2.legend()
# # # ax3.legend()

# # ax1.grid()
# # ax2.grid()

# # ax3.set_ylim([0,50]) 
# ax1.set_ylim(bottom=0)

# # ax3.set_ylabel('Covid-registrered deaths [%]')

# # if saveFigures:
# #     fig.savefig(f'Figures/ViggoMetode_{curAge}')

# %% [markdown]
# # Old below

# %%

# allMort,allMortDates = getAllCause(0,19)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curPop = getPopSizeNational(0,19)
# curRoll = [x.sum() for x in sliding_window_view(allCases_0_19, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_0_19 = allMort*curRollRela

# %%
# # getTimeSeriesAll()
# ssiAges = df2_C.Aldersgruppe.unique()
# print(ssiAges)
# firstAges = [int(x.split('-')[0]) for x in ssiAges[:-3]]
# firstAges.append(80)

# lastAges = [int(x.split('-')[1]) for x in ssiAges[:-3]]
# lastAges.append(125)
    
#     firstIndex = 0
#     lastIndex = -1

#     if (minAge == 95):
#         firstIndex = -2
#     elif (firstAge == minAge).any():
#         firstIndex = np.where(firstAge == minAge)[0][0]
#     else:
#         print('Incorrect minimum age, using 0')
        
#     if (maxAge == 125):
#         lastIndex = -1
#     elif (lastAge == maxAge).any():
#         lastIndex = np.where(lastAge == maxAge)[0][0]+1
#     else:
#         print('Incorrect maximum age, using maximum')


# %%
# allDeaths = getTimeSeries(df2_D,sNone)+getTimeSeries(df2_D,sOne)+getTimeSeries(df2_D,sTwo)
# allCases = getTimeSeries(df2_C,sNone)+getTimeSeries(df2_C,sOne)+getTimeSeries(df2_C,sTwo)

# # fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
# # ax1.plot(allDates,allCases)
# # ax2.plot(allDates,allDeaths)

# # ax1.set_xlim([np.datetime64('2021-10'),np.datetime64('2022-03')])
# # ax1.grid()
# # ax2.grid()
# # ax2.set_ylim(bottom=0)

# %%
# # get
# # allCases = getTimeSeries(df2_C,sNone)+getTimeSeries(df2_C,sOne)+getTimeSeries(df2_C,sTwo)
# allCases_0_19 = getTimeSeriesAll(df2_C,'0-5') +getTimeSeriesAll(df2_C,'6-11') +getTimeSeriesAll(df2_C,'12-15') +getTimeSeriesAll(df2_C,'16-19') 
# # allCases_0_19 = getTimeSeriesAll(df2_C,'0-5') +getTimeSeriesAll(df2_C,'6-11') +getTimeSeriesAll(df2_C,'12-15') +getTimeSeriesAll(df2_C,'16-19') 
# # allCases_0_19 = getTimeSeriesAll(df2_C,'0-5') +getTimeSeriesAll(df2_C,'6-11') +getTimeSeriesAll(df2_C,'12-15') +getTimeSeriesAll(df2_C,'16-19') 
# # allCases_0_19 = getTimeSeriesAll(df2_C,'0-5') +getTimeSeriesAll(df2_C,'6-11') +getTimeSeriesAll(df2_C,'12-15') +getTimeSeriesAll(df2_C,'16-19') 
# allCases_0_19

# allMort,allMortDates = getAllCause(0,19)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curPop = getPopSizeNational(0,19)
# curRoll = [x.sum() for x in sliding_window_view(allCases_0_19, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_0_19 = allMort*curRollRela

# %%
# df2_C

# %%

# allCases_20_39 = getTimeSeriesAll(df2_C,'20-29') +getTimeSeriesAll(df2_C,'30-39')
# allCases_20_39

# allMort,allMortDates = getAllCause(20,39)
# curPop = getPopSizeNational(20,39)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curRoll = [x.sum() for x in sliding_window_view(allCases_20_39, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_20_39 = allMort*curRollRela

# %%

# allCases_40_64 = getTimeSeriesAll(df2_C,'40-49') +getTimeSeriesAll(df2_C,'50-59') +getTimeSeriesAll(df2_C,'60-64')
# allCases_40_64

# allMort,allMortDates = getAllCause(40,64)
# curPop = getPopSizeNational(40,64)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curRoll = [x.sum() for x in sliding_window_view(allCases_40_64, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_40_64 = allMort*curRollRela

# %%

# allCases_65_79 = getTimeSeriesAll(df2_C,'65-69') +getTimeSeriesAll(df2_C,'70-79')
# allCases_65_79

# allMort,allMortDates = getAllCause(65,79)
# curPop = getPopSizeNational(65,79)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curRoll = [x.sum() for x in sliding_window_view(allCases_65_79, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_65_79 = allMort*curRollRela

# %%


# allCases_80 = getTimeSeriesAll(df2_C,'80+') 
# allCases_80

# allMort,allMortDates = getAllCause(80,125)
# curPop = getPopSizeNational(80,125)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curRoll = [x.sum() for x in sliding_window_view(allCases_80, window_shape = 4)]
# curRollRela = curRoll/curPop

# DeathRandom_80 = allMort*curRollRela

# %%

# fig,ax1 = plt.subplots()

# ax1.plot(allMortDates,DeathRandom_0_19)
# ax1.plot(allMortDates,DeathRandom_20_39)
# ax1.plot(allMortDates,DeathRandom_40_64)
# ax1.plot(allMortDates,DeathRandom_65_79)
# ax1.plot(allMortDates,DeathRandom_80)
# ax1.plot(allMortDates,DeathRandom_0_19+DeathRandom_20_39+DeathRandom_40_64+DeathRandom_65_79+DeathRandom_80)

# %%
# allMort,allMortDates = getAllCause(0,19)
# allMort = allMort[-13:]
# allMortDates = allMortDates[-13:]
# curPop = getPopSizeNational(0,19)

# allCases_0_19
# from numpy.lib.stride_tricks import sliding_window_view

# curRoll = [x.sum() for x in sliding_window_view(allCases_0_19, window_shape = 4)]
# curRollRela = curRoll/curPop

# # curRoll/curPop

# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

# ax1.plot(allDates[3:],curRollRela)
# ax2.plot(allMortDates,allMort)
# ax2.plot(allMortDates,allMort*curRollRela)

# %%
# allCases_0_19


