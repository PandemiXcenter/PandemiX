# %% [markdown]
# # Notebook for making a number of figures of the current status of herd immunity in Denmark
# Both distributed by age and municipality
# 
# Main figures to be made here:
# * Cumulative incidence since December 15th, curve on national level, by age
# * Cumulative incidence since December 15th, map on municipality level
# * Explanation of measure of epidemic progression (growing vs falling)
# * Map on municipality level of the above-mentioned measure
# 
# Questions about data that should be checked:
# In Kommune_DB/07_bekraeftede_tilfaelde_pr_dag_pr_kommune.csv, the readme does not accurately specify what the date is.
# Kommune_DB/17_tilfaelde_fnkt_alder_kommuner.csv appears to be based on testing date.
# 
# If 07_bekraeftede_tilfaelde_pr_dag_pr_kommune is based on opgørelsesdato, then there is no problem. If both are testing date, something is wrong with data or my method of collecting it... 
# 

# %% [markdown]
# # First, load various packages and set figure preferences

# %%

import matplotlib


# Load packages and settings
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)
import seaborn as sns

import geopandas as gpd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:dark yellow","gray","xkcd:green"],N=len(curRangeToShow))

import locale
import matplotlib.dates as mdates
locale.setlocale(locale.LC_TIME,"Danish")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
# ax1.spines['top'].set_visible(False) 

import os
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

path_figs = path_figs + "Immunity_SSI\\"

# %% [markdown]
# # Get latest data files 

# %%
# Municipality maps
gdf = gpd.read_file(rootdir_data+'Kommune\\Kommune.shp')

# Only use most recent mapdata
gdf = gdf[gdf.til == np.max(gdf.til.unique())]

# %%
# Municipality data, cumulative sums
latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
df_07 = pd.read_csv(latestdir+'/Kommunalt_DB/07_bekraeftede_tilfaelde_pr_dag_pr_kommune.csv',encoding='latin1',delimiter = ';')

df_07['Dato'] = pd.to_datetime(df_07['Dato'])

# %%
# Get file to relate kommunekode to names
latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                delimiter = ';')
df_kommunekort = df_kommunekort.set_index("Kommunenavn")

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

# %%
# Collect age-specific cases for all municipalities
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
# Age-specific number can be found in Regionalt_DB\\18_fnkt_alder_... file 
# However, using the sum of the data in dataframe df gives it on a weekday resolution rather 
# than just weekly. However, this is only reasonable when looking at the cumulative sum of cases
df_total = df.groupby(['Aldersgruppe','Dagsdato']).sum()

# %% [markdown]
# # Define helper functions

# %%
def getPopSize(kommuneNavn,minAlder=0,maxAlder=125):

    # Some names differ between SSI and Danmark Statistik
    if (kommuneNavn == 'Høje Tåstrup'):
        kommuneNavn = 'Høje-Taastrup'
    if (kommuneNavn == 'Århus'):
        kommuneNavn = 'Aarhus'
    if (kommuneNavn == 'Nordfyn'):
        kommuneNavn = 'Nordfyns' 
    if (kommuneNavn == 'Vesthimmerland'):
        kommuneNavn = 'Vesthimmerlands'

        
    return popdf[(popdf.Kommune == kommuneNavn) & (popdf.AlderKort >= minAlder) & (popdf.AlderKort <= maxAlder)].Antal.sum()
    

# %%

def getPopSizeNational(minAlder=0,maxAlder=125):
    popdf_nat
    return popdf_nat[(popdf_nat.AlderKort >= minAlder) & (popdf_nat.AlderKort <= maxAlder)].Antal.sum()

# print(getPopSizeNational(1,2))
# popdf_nat
    

# %%
def getKommuneCount(curKommune):
    kommune_df = df_07.loc[df_07["Kommunenavn"] == curKommune]
    
    curDays = kommune_df['Dato'].values
    antal_borgere = df_kommunekort["Antal borgere"][curKommune]
    curPerc = ((kommune_df['Bekræftede tilfælde']/antal_borgere)*100).values
    curCount = kommune_df['Bekræftede tilfælde'].values 
    
    # Cut out most recent data, since it has not yet been counted entirely
    indexToUse = curDays <= (np.datetime64(latestsubdir[-10:])-np.timedelta64(2,'D'))
    curCount = curCount[indexToUse]
    curPerc = curPerc[indexToUse]
    curDays = curDays[indexToUse]

    return curDays,curCount,curPerc

# curDays,curCount,curPerc = getKommuneCount('København')

# %%

def getProgressionMeasure(curDays,curCount):

    # Difference between dates and the corresponding week-day the week before
    weekDiff = curCount[7:] - curCount[:-7]

    # Measure 1: Weekly difference, relative to week before
    measure1 =  (weekDiff)/curCount[:-7]

    # Measure 2: Sign of relative difference, currently unused
    measure2 = np.sign(weekDiff)

    # Right-adjusted days
    measureDays = curDays[7:]

    return rnTime(measureDays,7),rnMean(measure1,7)

# getProgressionMeasure(curDays,curCount)

# %% [markdown]
# # Make figures
# ## First: Age-distribution, national level, curve

# %%
allAges = df.Aldersgruppe.unique()[:-1] # An agegroup called "." was sometimes included?
# firstDateToCount = np.datetime64('2021-10-01') # For comparison with older figures
firstDateToCount = np.datetime64('2021-12-15')
textFirstDate = pd.to_datetime(firstDateToCount).strftime('%#d. %B %Y')

fig,ax1 = plt.subplots(figsize=(13,6.5),tight_layout=True)

fig.patch.set_facecolor('xkcd:off white')
ax1.set_facecolor('xkcd:off white')

curMax = 0

for curAge in allAges:
    curDates = df_total.loc[curAge].index
    curCounts = df_total.loc[curAge,'Bekræftede tilfælde']
    
    # Count cumulative sum on firstDate
    initialCount = curCounts[curDates >= firstDateToCount].iloc[0]

    curCounts = curCounts-initialCount

    if (curAge == '80+'):
        curMinAge,curMaxAge = 80,125
    else:
        curMinAge,curMaxAge = [int(x) for x in curAge.split('-')]
    curPopSize = getPopSizeNational(curMinAge,curMaxAge)

    # ax1.plot(curDates,curCounts,'.-',label=curAge)
    ax1.plot(curDates,100*curCounts/curPopSize,label=curAge)

    curMax = np.max([curMax,np.max(100*curCounts/curPopSize)])

ax1.legend()
ax1.set_xlim(left=np.datetime64('2021-10-01'))

# Draw weekends
firstSunday = np.datetime64('2021-10-03')
numWeeks = 52
for k in range(0,numWeeks):
    curSunday = firstSunday + np.timedelta64(7*k,'D')
    ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')


ax1.set_ylabel(f'Kumulerede smittetilfælde siden {textFirstDate}\nAndel af borgere i aldersgruppen [%]')

curXticks = [np.datetime64(x) for x in [
    '2021-12-01',
    '2021-12-15',
    '2022-01-01',
    '2022-01-15',
    '2022-02-01',
    '2022-02-15',
    '2022-03-01',
    '2022-03-15',
    '2022-04-01',
    '2022-04-15',
]]
ax1.set_xticks(curXticks)


# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
ax1.set_xlim(left=firstDateToCount,right=np.datetime64('today')+np.timedelta64(5,'D'))
ax1.set_ylim(bottom=0)
ax1.set_ylim(top=np.round(curMax/5)*5+5)
# ax1.set_ylim(top=np.round(curMax/5)*5)
ax1.legend(loc='upper left')
ax1.grid(axis='y')

# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

if saveFigures:
    plt.savefig(path_figs+'NationaltKumuleret')

# %% [markdown]
# # Calculate measure and show example

# %%
kommunenavne = df_07.Kommunenavn.unique()


gdf_meas = gdf.copy()
# curDays,curCount,curPerc = getKommuneCount('København')


for kommunenavn in kommunenavne:
    curDays,curCount,curPerc = getKommuneCount(kommunenavn)

    measureDay,curMeasure = getProgressionMeasure(curDays,curCount)


    kommunenavnGdf = kommunenavn
    if (kommunenavn == 'Aabenraa'):
        kommunenavnGdf = 'Åbenrå'
    if (kommunenavn == 'Nordfyn'):
        kommunenavnGdf = 'Nordfyns'
    if (kommunenavn == 'København'):
        kommunenavnGdf = 'Københavns'
    if (kommunenavn == 'Bornholm'):
        kommunenavnGdf = 'Bornholms'
    if (kommunenavn == 'Faaborg-Midtfyn'):
        kommunenavnGdf = 'Fåborg-Midtfyn'
    if (kommunenavn == 'Lyngby-Taarbæk'):
        kommunenavnGdf = 'Lyngby-Tårbæk'

    curMeasureValue = curMeasure[-1]
    gdf_meas.loc[gdf_meas.navn == (kommunenavnGdf+' Kommune'),'CurrentMeasure'] = curMeasureValue
    # curPerc[curDays >= firstDateToCount]
    immuOnFirstDate = np.cumsum(curPerc)[curDays >= firstDateToCount][0]
    curTotImmu = np.cumsum(curPerc)[-1] - immuOnFirstDate
    gdf_meas.loc[gdf_meas.navn == (kommunenavnGdf+' Kommune'),'TotalImmunity'] = curTotImmu
# gdf_meas


# Clip measure outside -1 and 1
gdf_meas['CurrentMeasure'] = np.clip(gdf_meas.CurrentMeasure.values,-1,1)

# %%

# def getProgressionMeasure(curDays,curCount):

#     # Difference between dates and the corresponding week-day the week before
#     weekDiff = curCount[7:] - curCount[:-7]

#     # Measure 1: Weekly difference, relative to week before
#     measure1 =  (weekDiff)/curCount[:-7]

#     # Measure 2: Sign of relative difference, currently unused
#     measure2 = np.sign(weekDiff)

#     # Right-adjusted days
#     measureDays = curDays[7:]

#     return rnTime(measureDays,7),rnMean(measure1,7)

kommunenavn = 'København'
kommunenavn = 'Odense'
curDays,curCount,curPerc = getKommuneCount(kommunenavn)

weekDiff = curCount[7:] - curCount[:-7]

# Measure 1: Weekly difference, relative to week before
measure1 =  (weekDiff)/curCount[:-7]

# Measure 2: Sign of relative difference, currently unused
measure2 = np.sign(weekDiff)

# Right-adjusted days
measureDays = curDays[7:]

fig,ax1 = plt.subplots(tight_layout=True)

fig.patch.set_facecolor('xkcd:off white')
ax1.set_facecolor('xkcd:off white')

ax1.plot(curDays,curCount,'*:k',linewidth=1,markersize=10)

# Draw weekends
firstSunday = np.datetime64('2021-10-03')
numWeeks = 52
for k in range(-numWeeks,numWeeks):
        curSunday = firstSunday + np.timedelta64(7*k,'D')
        ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
# ax1.grid(axis='y')

NotYetShown_down = True
NotYetShown_up = True
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

curSum = 0

curAvgVal = 0
# for i in range(0,8):
for i in range(6,-1,-1):

    curD1 = curDays[-7-1-i]
    curD2 = curDays[-1-i]
    curY1 = curCount[-7-1-i]
    curY2 = curCount[-1-i]

    curMeas = measure1[-1-i]
    curSum = curSum + curMeas
    curAvgVal = curAvgVal+curCount[-1-i]

    dY = curY2-curY1

    if (dY >= 0):
        curColor = 'r'
        curLabel = NotYetShown_up * 'Vækst'
        ax1.plot([curD1,curD2],[curY1,curY2],'--',color=curColor,label=curLabel)
        # ax1.text(curD2,curY2,f'Stigning på {curMeas:0.2}')
        # ax1.text(curD2,curY2+50,f'{curMeas:+0.2}',ha='center',bbox=props)
        ax1.text(curD2,curY2+50,f'{100*curMeas:+3.0f}%',ha='center',bbox=props)

        NotYetShown_up = False 
    else:
        curColor = 'b'
        curLabel = NotYetShown_down * 'Fald'
        ax1.plot([curD1,curD2],[curY1,curY2],'--',color=curColor,label=curLabel)
        NotYetShown_down = False
        # ax1.text(curD2,curY2+50,f'{curMeas:+0.2}',ha='center',bbox=props)
        ax1.text(curD2,curY2+50,f'{100*curMeas:+3.0f}%',ha='center',bbox=props)
    
curAvgVal = curAvgVal/7
# ax1.text(np.datetime64('today')-np.timedelta64(1,'D'),curAvgVal,f'Sum: {7*rnMean(measure1,7)[-1]:0.2}',fontsize=20, weight='bold')
ax1.text(np.datetime64('today')-np.timedelta64(1,'D'),curAvgVal,f'Sum: {100*7*rnMean(measure1,7)[-1]:3.0f}',fontsize=20, weight='bold')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
# ax1.plot(measureDays,weekDiff)
leftDate = np.datetime64('2022-01-14')
ax1.set_xlim(left=leftDate,right=np.datetime64('today')+np.timedelta64(4,'D'))
ax1.set_ylim(bottom=1000)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_ylabel('Antal nye smittetilfælde')
ax1.set_title(f'Eksempel på mål. Kommune: {kommunenavn}')

if saveFigures:
    plt.savefig(path_figs+f'MeasureExample')

# %%
rnMean(measure1,7)[-4:]*7

# %%
# Example of measure for different places


ExamplesToShow = ['København','Århus','Aalborg','Horsens','Roskilde','Holbæk','Odense']

# fig,allAxes = plt.subplots(2,2)


for i in range(len(ExamplesToShow)):
    kommunenavn = ExamplesToShow[i]
    #     curDays,curCount,curPerc = getKommuneCount(kommunenavn)

    #     measureDay,curMeasure = getProgressionMeasure(curDays,curCount)

    cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:blue","xkcd:light blue","xkcd:light gray","xkcd:red","xkcd:dark red"],N=11)
    vmin = -0.5
    vmax = 0.5
    rangeToShow = np.linspace(vmin,vmax,11)

    leftDate = np.datetime64('2021-11-01')

    # fig,ax2 = plt.subplots()
    # ax2.imshow([[-0.5,-0.5],[0.5,0.5]],cmap=cmap,interpolation='bicubic',vmin=-0.5,vmax=0.5)

    # kommunenavn = 'København'
    curDays,curCount,curPerc = getKommuneCount(kommunenavn)
    measureDay,curMeasure = getProgressionMeasure(curDays,curCount)

    fig,(ax1,ax2) = plt.subplots(2,1,tight_layout=True,sharex=True,gridspec_kw={'height_ratios': [3,1]})

    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')
    ax1.plot(curDays,curCount,'k.:',linewidth=0.5,markersize=2,label='Data')
    ax1.plot(rnTime(curDays,7),rnMean(curCount,7),'k',label='7-dages gennemsnit')


    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(0,numWeeks):
        curSunday = firstSunday + np.timedelta64(7*k,'D')
        ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')

    # ax2.imshow([[-0.5,-0.5],[0.5,0.5]],cmap=cmap,interpolation='bicubic',vmin=-0.5,vmax=0.5)
    import matplotlib.patches as patches 

    rect = patches.Rectangle((leftDate,-1),np.timedelta64(30*10,'D'),1,facecolor=cmap(0))
    ax2.add_patch(rect)
    rect = patches.Rectangle((leftDate,0),np.timedelta64(30*10,'D'),1,facecolor=cmap(11))
    ax2.add_patch(rect)
    for curVal in rangeToShow:
        rect = patches.Rectangle((leftDate,curVal-0.05),np.timedelta64(30*10,'D'),0.1,facecolor=cmap(curVal+0.5))
        ax2.add_patch(rect)
    ax2.axhline(0,ls='--',color='k',linewidth=2)
    ax2.plot(measureDay,curMeasure,linewidth=5,color='k')
    ax2.plot(measureDay,curMeasure,linewidth=2,color='xkcd:lavender')
    # ax2.plot(measureDay,curMeasure,color='xkcd:dark purple')
    # ax2.plot(measureDay,curMeasure,':',color='xkcd:neon green')
    # ax2.plot(measureDay,np.zeros(curMeasure.shape),'k:')

    # ax2.imshow([[-0.5,-0.5],[0.5,0.5]],cmap=cmap,interpolation='bicubic',vmin=-0.5,vmax=0.5)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim([-1,1])
    ax1.set_xlim(left=leftDate,right=np.datetime64('today')+np.timedelta64(7,'D'))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    ax1.set_ylabel('Antal nye smittetilfælde')
    ax2.set_ylabel('Mål for udvikling')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.legend()
    ax1.set_title(kommunenavn)

    if saveFigures:
        plt.savefig(path_figs+f'ByEksempler_{kommunenavn}')

# %%
# # # gdf_meas.CurrentMeasure.values
# # # np.clip(gdf_meas.CurrentMeasure.values,-1,1)

# # gdf_meas["CurrentMeasure_Removed"] = gdf_meas.CurrentMeasure
# # gdf_meas.loc[gdf_meas.navn=='Samsø Kommune',"CurrentMeasure_Removed"] = np.nan
# # gdf_meas.loc[gdf_meas.navn=='Læsø Kommune',"CurrentMeasure_Removed"] = np.nan

# # gdf_meas_noNaN = gdf_meas.dropna()
# # # gdf_meas_noNaN
# # gdf_meas[gdf_meas.CurrentMeasure.isna()]
# # np.array(curPerc)[curDates >= firstDateToCount]
# # curPerc
# # curPerc[np.array(curDates) >= firstDateToCount]
# curDays,curCount,curPerc = getKommuneCount(kommunenavn)
# len(curPerc)
# # len(curDates)

# # curPerc
# # curDates >= firstDateToCount
# # curTotImmu

# immuOnFirstDate = curPerc[curDays >= firstDateToCount][0]
# immuOnFirstDate
# curTotImmu = np.cumsum(curPerc)[-1] - immuOnFirstDate
# curTotImmu
# gdf_meas

# %% [markdown]
# # Second figure: Map of measure

# %%

fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size="5%", pad=0.01)
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cax = inset_axes(ax1, width="30%", height="30%", loc=3) 

fig.patch.set_facecolor('xkcd:beige')
ax1.set_facecolor('xkcd:beige')

vmin = -0.5
vmax = 0.5
# vmin=-1
# vmax=1


# cmap = plt.cm.get_cmap('seismic')
# cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:blue","xkcd:light gray","xkcd:dark red"],N=13)
cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:blue","xkcd:light blue","xkcd:light gray","xkcd:red","xkcd:dark red"],N=11)
# gdf_meas.plot(ax=ax1,column = 'CurrentMeasure',cmap=cmap,legend=True, cax=cax)
gdf_meas.plot(  ax=ax1,
                column = 'CurrentMeasure',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolor="black",
                linewidth=0.2,
                legend=True,
                # legend_kwds={'loc': 'lower right'},
                # legend_kwds={'loc': 'lower right'},
                cax=cax
                )

                

# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

cax.yaxis.set_label_position("left")
cax.yaxis.tick_left()
# cax.set_yticks([])

# curYticks = [-1,-5/11,0,5/11,1]
curYticks = np.array([-1,-5/11,0,5/11,1])*vmax
# curYtickLabels = ['Stort fald','Fald','Uændret','Stigning','Stor stigning']
curYtickLabels = ['Stort\nfald','Fald','Uændret','Stigning','Stor\nstigning']
cax.set_yticks(curYticks)
cax.set_yticklabels(curYtickLabels)

if saveFigures:
    plt.savefig(path_figs+'KortSmitteudvikling')


# %% [markdown]
# # Cumulative incidence

# %%

fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size="5%", pad=0.01)
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cax = inset_axes(ax1, width="30%", height="30%", loc=3) 

fig.patch.set_facecolor('xkcd:beige')
ax1.set_facecolor('xkcd:beige')

vmin = 10
vmax= 30
dY = 2.5
curYticks = np.arange(vmin,vmax+dY,dY)

# cmap = plt.cm.get_cmap('seismic')
# cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:blue","xkcd:light gray","xkcd:dark red"],N=13)
# cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:blue","xkcd:light blue","xkcd:light gray","xkcd:red","xkcd:dark red"],N=11)
cmap = colors.LinearSegmentedColormap.from_list("", ["xkcd:purple","xkcd:light gray","xkcd:orange"],N=len(curYticks)-1)
# gdf_meas.plot(ax=ax1,column = 'CurrentMeasure',cmap=cmap,legend=True, cax=cax)
gdf_meas.plot(  ax=ax1,
                column = 'TotalImmunity',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolor="black",
                linewidth=0.2,
                legend=True,
                # legend_kwds={'loc': 'lower right'},
                # legend_kwds={'loc': 'lower right'},
                cax=cax
                )

                

# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

cax.yaxis.set_label_position("left")
cax.yaxis.tick_left()
# cax.set_yticks([])

# curYticks = [-1,-5/11,0,5/11,1]
# curYtickLabels = ['Stort fald','Fald','Uændret','Stigning','Stor stigning']
# cax.set_yticks(curYticks)
# cax.set_yticklabels(curYtickLabels)

cax.set_yticks(curYticks)
# plt.show()
newYticksLabels = [str(np.round(x,1)).replace('.',',') for x in curYticks]
# print(newYticksLabels)
cax.set_yticklabels(newYticksLabels)
# cax.set_yticklabels(asdf)

ax1.set_title('Naturlig immunitet fra Omikron smitte, samlet befolkning')
cax.yaxis.set_label_position("left")
cax.yaxis.tick_left()
textFirstDate = pd.to_datetime(firstDateToCount).strftime('%#d. %B %Y')
lastDateUsed = np.datetime64(latestsubdir[-10:])-np.timedelta64(1,'D')
textLastDate = pd.to_datetime(lastDateUsed).strftime('%#d. %B %Y')
cax.set_ylabel(f'Andel af befolkning smittet siden {textFirstDate} [%]')
ax1.set_title(f'Andel af befolkning smittet i perioden {textFirstDate} til og med {textLastDate}\nBaseret på PCR-positive, ikke justeret for test-intensitet')



if saveFigures:
    plt.savefig(path_figs+'KortImmunitet')


# plt.show()

# %% [markdown]
# # Various tests, not used

# %%

# latestsubdir = list(os.walk(path_dash))[0][1][-1]
# latestdir = path_dash + latestsubdir
# latestdir

# dfAge = pd.read_csv(latestdir+'\\Regionalt_DB\\18_fnkt_alder_uge_testede_positive_nyindlagte.csv',delimiter=';',encoding='latin1',dtype=str)
# dfAge['Nyindlagte pr. 100.000 borgere'] = pd.to_numeric(dfAge['Nyindlagte pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Positive pr. 100.000 borgere'] = pd.to_numeric(dfAge['Positive pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Testede pr. 100.000 borgere'] = pd.to_numeric(dfAge['Testede pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Antal testede'] = pd.to_numeric(dfAge['Antal testede'])
# dfAge['Antal positive'] = pd.to_numeric(dfAge['Antal positive'])

# # dfAge.tail(18)

# %%
# # Calculate dates
# import datetime
# weekDTs = [np.datetime64(datetime.datetime.strptime(d[:4] + '-W'+d[6:8]+'-1', "%Y-W%W-%w")) for d in dfAge.Uge]
# dfAge['Dato'] = weekDTs

# # # Remove everything before 2021-01-01
# # dfAge = dfAge[dfAge.Dato > np.datetime64('2020-12-31')]
# dfAge.columns
# dfAge.Aldersgruppe.unique()

# %%
# dfAge.columns
# # dfAge

# %%
# df_total = df.groupby(['Aldersgruppe','Dagsdato']).sum()

# plt.figure()
# plt.plot(df_total.loc['0-2'].index,df_total.loc['0-2','Bekræftede tilfælde'],'*:')
# # plt.plot(df_total.loc['0-2'].index,df_total.loc['0-2','Bekræftede tilfælde'].diff())
# # plt.plot(rnTime(df_total.loc['0-2'].index,7),rnMean(df_total.loc['0-2','Bekræftede tilfælde'].diff(),7))
# plt.plot(df_total.loc['20-39'].index,df_total.loc['20-39','Bekræftede tilfælde'],'*:')

# curdfAge = dfAge[dfAge.Aldersgruppe == '00-02']
# plt.plot(curdfAge.Dato,np.cumsum(curdfAge['Antal positive']),'o-')
# # plt.plot(curdfAge.Dato,curdfAge['Antal positive'],'.')

# curdfAge = dfAge[dfAge.Aldersgruppe == '20-39']
# plt.plot(curdfAge.Dato,np.cumsum(curdfAge['Antal positive']),'o-')

# plt.xlim(left=np.datetime64('2021-10-01'))

# allAges = df.Aldersgruppe.unique()[:-1] # An agegroup called "." was sometimes included?
# allAges
# # df_07

# %%
# # Get data of age-specific cases
# latestsubdir = list(os.walk(path_dash))[0][1][-1]
# latestdir = path_dash + latestsubdir
# latestdir

# dfAge = pd.read_csv(latestdir+'\\Regionalt_DB\\18_fnkt_alder_uge_testede_positive_nyindlagte.csv',delimiter=';',encoding='latin1',dtype=str)
# dfAge['Nyindlagte pr. 100.000 borgere'] = pd.to_numeric(dfAge['Nyindlagte pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Positive pr. 100.000 borgere'] = pd.to_numeric(dfAge['Positive pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Testede pr. 100.000 borgere'] = pd.to_numeric(dfAge['Testede pr. 100.000 borgere'].str.replace(',','.'))
# dfAge['Antal testede'] = pd.to_numeric(dfAge['Antal testede'])
# dfAge['Antal positive'] = pd.to_numeric(dfAge['Antal positive'])

# dfAge.tail(18)


# %%
# # df_07.tail()
# # # df_07[df_07.Kommune == 101]
# # Municipality data, totals
# latestsubdir = list(os.walk(path_dash))[0][1][-1]
# latestdir = path_dash + latestsubdir
# df_07 = pd.read_csv(latestdir+'/Kommunalt_DB/07_bekraeftede_tilfaelde_pr_dag_pr_kommune.csv',encoding='latin1',delimiter = ';')

# df_07['Dato'] = pd.to_datetime(df_07['Dato'])

# %%

# plt.figure()
# curdf = asdf.loc[101]
# plt.plot(curdf.index,curdf['Bekræftede tilfælde'])

# curdf1 = df_07[df_07.Kommune == 101]

# plt.plot(curdf1.Dato,np.cumsum(curdf1['Bekræftede tilfælde']))
# # plt.plot(rnTime(curdf1.Dato,7),rnMean(curdf1['Bekræftede tilfælde'],7))
# # df[df.Kommune == 101]

# plt.xlim(left=np.datetime64('2021-10-01'))

# %%
# df_07
# asdf = df.groupby(['Kommune','Dagsdato']).sum()

# asdf.loc[101]

# plt.figure()
# curdf = asdf.loc[101]
# plt.plot(curdf.index,curdf['Bekræftede tilfælde'].diff(),'b.:',linewidth=1)
# plt.plot(rnTime(curdf.index,7),rnMean(curdf['Bekræftede tilfælde'].diff(),7),'b')

# curdf1 = df_07[df_07.Kommune == 101]

# plt.plot(curdf1.Dato,curdf1['Bekræftede tilfælde'],'r.:',linewidth=1)
# plt.plot(rnTime(curdf1.Dato,7),rnMean(curdf1['Bekræftede tilfælde'],7),'r')
# # df[df.Kommune == 101]

# plt.xlim(left=np.datetime64('2021-10-01'))

# %%
# df.tail()
# df.groupby('Dagsdato').sum()

# %%
# plt.figure()
# curdf = df_07.groupby('Dato').sum()

# # plt.plot(curdf.index,curdf['Bekræftede tilfælde'])
# plt.plot(curdf.index,np.cumsum(curdf['Bekræftede tilfælde']))

# curdf1 = df.groupby('Dagsdato').sum()
# plt.plot(curdf1.index,curdf1['Bekræftede tilfælde'],':')

# plt.xlim(left=np.datetime64('2021-10-01'))

# %%


plt.close('all')




print('Finished making all figures')