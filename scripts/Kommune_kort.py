# %%

# %matplotlib widget
# Load packages and settings
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)
import seaborn as sns

import geopandas as gpd

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# path_figs = path_figs + "Immunity_SSI\\"

# %%
gdf = gpd.read_file(rootdir_data+'Kommune\\Kommune.shp')

# Only use most recent mapdata
gdf = gdf[gdf.til == np.max(gdf.til.unique())]

# %%
df_meas = pd.read_csv('KommuneMeasure.csv')

# %%
# Test of colorrange
posVals = np.arange(-7,8,2)

import matplotlib
cmap = matplotlib.cm.get_cmap('Spectral')
[cmap(x/7) for x in posVals]

cmap = matplotlib.cm.get_cmap('Spectral',len(posVals))

cmap(7)
posVals 
df_meas.iloc[:,1:].transpose().iloc[:,0].values

# %%
allKoms = df_meas.columns[1:]
# [x.split(' Kommun')[0] for x in gdf.navn]

fig,ax1 = plt.subplots() 
# gdf.plot(color='gray',ax=ax1)

# for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf.navn):
#     ax1.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points")

cmap = matplotlib.cm.get_cmap('seismic')
# cmap = matplotlib.cm.get_cmap('seismic')

for curKom in allKoms:
    # curKom = 'Horsens'

        
    curVal = df_meas[curKom].values[0]
    curVal = (1+df_meas[curKom].values[0])/2
    # curVal = (1+2*df_meas[curKom].values[0])/2
    # if (curVal >= 0):
    #     curColor = [curVal,0.1,0]
    # else:
    #     curColor = [-curVal/4,-curVal/4,-curVal]
    # # curColor = [0.5+curVal/2,0,0.5-curVal/2]

    # [cmap(x/7) for x in posVals]
    curColor = cmap(curVal)
    
    if (curKom == 'Aabenraa'):
        curKom = 'Åbenrå'
    if (curKom == 'Nordfyn'):
        curKom = 'Nordfyns'
    if (curKom == 'København'):
        curKom = 'Københavns'
    if (curKom == 'Bornholm'):
        curKom = 'Bornholms'
    if (curKom == 'Faaborg-Midtfyn'):
        curKom = 'Fåborg-Midtfyn'
    if (curKom == 'Lyngby-Taarbæk'):
        curKom = 'Lyngby-Tårbæk'
        

    curgdf = gdf[gdf.navn == (curKom+' Kommune')]
    curgdf.plot(ax=ax1,color=curColor)

# Hand-crafted legend
import matplotlib.patches as patches
# posVals = np.arange(-7,8,2)
posVals = np.arange(-1,1.1,2/7)
posValsStr = [str(int(np.round(7*x))) for x in posVals]
posValsStr = [
    '7 dages vækst',
    '6 dages vækst, 1 dags fald',
    '5 dages vækst, 2 dages fald',
    '4 dages vækst, 3 dages fald',
    '3 dages vækst, 4 dages fald',
    '2 dages vækst, 5 dages fald',
    '1 dags vækst, 6 dages fald',
    '7 dages fald',
]

for i in range(len(posVals)):
    curVal = posVals[i]
    curVal = 1-(1+posVals[i])/2
    curColor = cmap(curVal)
    curStr = posValsStr[i]
    x0 = 680000
    y0 = 6.35e6
    wi = 9000
    he = 10000
    cury = y0-he*i -he*0.5*i
    rect = patches.Rectangle((x0,cury),wi,he,linewidth=2,edgecolor='k',facecolor=curColor)
    ax1.add_patch(rect)
    # ax1.annotate('asdf',xy=(x0,y0))
    ax1.text(x0+wi*2,cury,curStr,ha='left')

# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

ax1.set_title('')

if saveFigures:
    fig.savefig(path_figs+'/Maps/KommuneUdvikling')
    curDate = np.datetime64('today')
    fig.savefig(path_figs+f'/Maps/KommuneUdvikling_{curDate}')

# ax1.set_xlim([690000,750000])
# ax1.set_ylim([6.16e6,6.195e6])


# for i in range(len(posVals)):
#     curVal = posVals[i]
#     curVal = 1-(1+posVals[i])/2
#     curColor = cmap(curVal)
#     curStr = posValsStr[i]
#     x0 = 735000
#     y0 = 6.18e6
#     wi = 9000/4
#     he = 10000/6
#     cury = y0-he*i -he*0.5*i
#     rect = patches.Rectangle((x0,cury),wi,he,linewidth=2,edgecolor='k',facecolor=curColor)
#     ax1.add_patch(rect)
#     # ax1.annotate('asdf',xy=(x0,y0))
#     ax1.text(x0+wi*2,cury,curStr,ha='left')

# if saveFigures:
#     fig.savefig(path_figs+'/Maps/KommuneUdvikling_Hovedstaden')

# %%
np.sort(allKoms)

# %% [markdown]
# # Make a new figure, but with explanations
# First, method from Kommune_Progression is repeated

# %%

# Walk to relavant folder
latestsubdir = list(os.walk(path_dash))[0][1][-1]
latestdir = path_dash + latestsubdir
df_07 = pd.read_csv(latestdir+'/Kommunalt_DB/07_bekraeftede_tilfaelde_pr_dag_pr_kommune.csv',encoding='latin1',delimiter = ';')

df_07['Dato'] = pd.to_datetime(df_07['Dato'])

# %%
def getKommuneCount(curKommune):
    kommune_df = df_07.loc[df_07["Kommunenavn"] == curKommune]
    
    # Cut out last data point
    kommune_df = kommune_df.iloc[:-1,:]
    # firstDate = np.datetime64(kommune_df.loc[kommune_df.index[0],'Dato'])-np.timedelta64(1,'D')
    # firstDate = np.datetime64('2021-11-01')
    # lastDate = np.datetime64(kommune_df.loc[kommune_df.index[-1],'Dato'])
    # Find number of citizens in region
    latestsubdir = list(os.walk(path_dash))[0][1][-1]
    latestdir = path_dash + latestsubdir
    df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                 delimiter = ';')
    df_kommunekort = df_kommunekort.set_index("Kommunenavn")

    # kommune_nr = kommune_df.Kommune.iloc[0]
    # kommune_df['Procent andel smittede'] = (kommune_df['Bekræftede tilfælde']/antal_borgere(curKommune))*100

    curDays = kommune_df['Dato'].values
    antal_borgere = df_kommunekort["Antal borgere"][curKommune]
    curPerc = ((kommune_df['Bekræftede tilfælde']/antal_borgere)*100).values
    curCount = kommune_df['Bekræftede tilfælde'].values 
    
    indexToUse = curDays <= (np.datetime64(latestsubdir[-10:])-np.timedelta64(2,'D'))
    curCount = curCount[indexToUse]
    curPerc = curPerc[indexToUse]
    curDays = curDays[indexToUse]

    return curDays,curCount,curPerc


# %%
def makeSmallPlot(ax1,curKommuneNavn):
    curDays,curCount,curPerc = getKommuneCount(curKommuneNavn)
    firstDateToShow = np.datetime64('today') - np.timedelta64(20,'D')
    firstIndex = np.where(curDays == firstDateToShow)[0][0]
    ax1.plot(curDays[firstIndex:],curCount[firstIndex:],'k*:',linewidth=1,label='Data')
    # ax1.plot(rnTime(curDays[firstIndex-4:],7),rnMean(curCount[firstIndex-4:],7),'k',linewidth=3,label='7-dages gennemsnit')

    i = 0

    NotYetShown_down = True
    NotYetShown_up = True

    for i in range(0,8):

        curD1 = curDays[-7-1-i]
        curD2 = curDays[-1-i]
        curY1 = curCount[-7-1-i]
        curY2 = curCount[-1-i]

        dY = curY2-curY1

        if (dY >= 0):
            curColor = 'r'
            curLabel = NotYetShown_up * 'Vækst'
            ax1.plot([curD1,curD2],[curY1,curY2],color=curColor,label=curLabel)

            NotYetShown_up = False 
        else:
            curColor = 'b'
            curLabel = NotYetShown_down * 'Fald'
            ax1.plot([curD1,curD2],[curY1,curY2],color=curColor,label=curLabel)
            NotYetShown_down = False


        # ax1.plot([curDays[-7-1-i],curDays[-1-i]],[curCount[-1-7-i],curCount[-1-i]])

        # ax1.set_ylim(bottom=0)

    ax1.set_xlim(left=firstDateToShow)
    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
            curSunday = firstSunday + np.timedelta64(7*k,'D')
            ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    # ax1.grid(axis='y')

    ax1.plot(curDays[firstIndex:],curCount[firstIndex:],'k*:',linewidth=1)

    ax1.legend(loc='upper left')

    ax1.set_ylabel('Antal smittetilfælde')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
    ax1.set_title(curKommuneNavn)


fig,ax1 = plt.subplots()

curKommuneNavn = 'København'
makeSmallPlot(ax1,curKommuneNavn)

fig.savefig(path_figs+'Maps/MeasureExample')

# %%
def makeSmallPlotSmall(ax1,curKommuneNavn):
    curDays,curCount,curPerc = getKommuneCount(curKommuneNavn)
    firstDateToShow = np.datetime64('today') - np.timedelta64(20,'D')
    firstIndex = np.where(curDays == firstDateToShow)[0][0]
    ax1.plot(curDays[firstIndex:],curCount[firstIndex:],'k*:',linewidth=1,markersize=2,label='Data')
    # ax1.plot(rnTime(curDays[firstIndex-4:],7),rnMean(curCount[firstIndex-4:],7),'k',linewidth=3,label='7-dages gennemsnit')

    i = 0

    NotYetShown_down = True
    NotYetShown_up = True

    for i in range(0,8):

        curD1 = curDays[-7-1-i]
        curD2 = curDays[-1-i]
        curY1 = curCount[-7-1-i]
        curY2 = curCount[-1-i]

        dY = curY2-curY1

        if (dY >= 0):
            curColor = 'r'
            curLabel = NotYetShown_up * 'Vækst'
            ax1.plot([curD1,curD2],[curY1,curY2],linewidth=2,color=curColor,label=curLabel)

            NotYetShown_up = False 
        else:
            curColor = 'b'
            curLabel = NotYetShown_down * 'Fald'
            ax1.plot([curD1,curD2],[curY1,curY2],linewidth=2,color=curColor,label=curLabel)
            NotYetShown_down = False


        # ax1.plot([curDays[-7-1-i],curDays[-1-i]],[curCount[-1-7-i],curCount[-1-i]])

        # ax1.set_ylim(bottom=0)

    ax1.set_xlim(left=firstDateToShow)
    # Draw weekends
    firstSunday = np.datetime64('2021-10-03')
    numWeeks = 52
    for k in range(-numWeeks,numWeeks):
            curSunday = firstSunday + np.timedelta64(7*k,'D')
            ax1.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    # ax1.grid(axis='y')

    ax1.plot(curDays[firstIndex:],curCount[firstIndex:],'k*:',linewidth=1,markersize=6)

    # ax1.legend(loc='upper left',fontsize=6)

    # ax1.set_ylabel('Antal smittetilfælde')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))

    textFontSize= 6
    ax1.set_title(curKommuneNavn,fontsize=textFontSize+4)
    ax1.tick_params(axis='both', which='major', labelsize=textFontSize)
    ax1.tick_params(axis='both', which='minor', labelsize=textFontSize)
    # ax1.set_xticklabels(ax1.get_xticklabels(),fontsize=6)

    curYlim = ax1.get_ylim()
    ax1.set_ylim([curYlim[0]*0.8,curYlim[1]*1.2])


fig,ax1 = plt.subplots()

curKommuneNavn = 'København'
makeSmallPlotSmall(ax1,curKommuneNavn)


# %%


# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

allKoms = df_meas.columns[1:]
# [x.split(' Kommun')[0] for x in gdf.navn]

fig,ax1 = plt.subplots(figsize=(15,15)) 
# gdf.plot(color='gray',ax=ax1)

# for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf.navn):
#     ax1.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points")

cmap = matplotlib.cm.get_cmap('seismic')

for curKom in allKoms:
    # curKom = 'Horsens'

        
    curVal = df_meas[curKom].values[0]
    curVal = (1+df_meas[curKom].values[0])/2
    # if (curVal >= 0):
    #     curColor = [curVal,0.1,0]
    # else:
    #     curColor = [-curVal/4,-curVal/4,-curVal]
    # # curColor = [0.5+curVal/2,0,0.5-curVal/2]

    # [cmap(x/7) for x in posVals]
    curColor = cmap(curVal)
    
    if (curKom == 'Aabenraa'):
        curKom = 'Åbenrå'
    if (curKom == 'Nordfyn'):
        curKom = 'Nordfyns'
    if (curKom == 'København'):
        curKom = 'Københavns'
    if (curKom == 'Bornholm'):
        curKom = 'Bornholms'
    if (curKom == 'Faaborg-Midtfyn'):
        curKom = 'Fåborg-Midtfyn'
    if (curKom == 'Lyngby-Taarbæk'):
        curKom = 'Lyngby-Tårbæk'
        

    curgdf = gdf[gdf.navn == (curKom+' Kommune')]
    curgdf.plot(ax=ax1,color=curColor)

# Hand-crafted legend
import matplotlib.patches as patches
# posVals = np.arange(-7,8,2)
posVals = np.arange(-1,1.1,2/7)
posValsStr = [str(int(np.round(7*x))) for x in posVals]
posValsStr = [
    '7 dages vækst',
    '6 dages vækst, 1 dags fald',
    '5 dages vækst, 2 dages fald',
    '4 dages vækst, 3 dages fald',
    '3 dages vækst, 4 dages fald',
    '2 dages vækst, 5 dages fald',
    '1 dags vækst, 6 dages fald',
    '7 dages fald',
]

for i in range(len(posVals)):
    curVal = posVals[i]
    curVal = 1-(1+posVals[i])/2
    curColor = cmap(curVal)
    curStr = posValsStr[i]
    # x0 = 680000
    # y0 = 6.35e6
    x0 = 630000
    y0 = 6.4e6
    wi = 9000
    he = 10000
    cury = y0-he*i -he*0.5*i
    rect = patches.Rectangle((x0,cury),wi,he,linewidth=2,edgecolor='k',facecolor=curColor)
    ax1.add_patch(rect)
    # ax1.annotate('asdf',xy=(x0,y0))
    ax1.text(x0+wi*2,cury,curStr,ha='left')

# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Make inset plots with description
insetW = 50000
insetH = 0.07e6

curKom = 'Albertslund'
curx0 = 750000
cury0 = 6.2e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+' Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')


curKom = 'København'
curx0 = 750000
cury0 = 6.1e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+'s Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')


curKom = 'Guldborgsund'
curx0 = 730000
cury0 = 6e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+' Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')

curKom = 'Aalborg'
curx0 = 420000
cury0 = 6.35e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
# axins.set_facecolor('xkcd:off white')
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
# bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
# plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+' Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')



curKom = 'Vejle'
curx0 = 330000
cury0 = 6.23e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
# axins.set_facecolor('xkcd:off white')
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
# bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
# plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+' Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')


curKom = 'Kolding'
curx0 = 330000
cury0 = 6.1e6
                #    bbox_to_anchor=(680000,6.35e6,685000,6.5e6),
axins = inset_axes(ax1, width=2, height=1,
                   bbox_to_anchor=(curx0,cury0,curx0+insetW,cury0+insetH),
                   bbox_transform=ax1.transData, loc=3)
# axins.set_facecolor('xkcd:off white')
makeSmallPlotSmall(axins,curKom)
bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(axins.get_xticklabels(), bbox=bbox)
plt.setp(axins.get_yticklabels(), bbox=bbox)
# bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
# plt.setp(axins.get_yticklabels(), bbox=bbox)
curgdf = gdf[gdf.navn == (curKom+' Kommune')]
ax1.plot([curx0 + insetW/2,curgdf.geometry.centroid.x ],[cury0 + insetH / 2, curgdf.geometry.centroid.y ],'k')


fig.patch.set_facecolor('xkcd:off white')
ax1.set_facecolor('xkcd:off white')
ax1.set_xlim(left=350000)

if saveFigures:
    fig.savefig(path_figs+'/Maps/KommuneUdvikling_Forklaring')


# %%
curgdf.geometry

# %% [markdown]
# # Map of local immunity

# %%
curgdf = gdf.copy()
curgdf['TotalImmunity'] = np.zeros((len(curgdf,)))
curgdf.head()

# %%

# fig,ax1 = plt.subplots(figsize=(15,15)) 

# for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf.navn):
#     ax1.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points")

# cmap = matplotlib.cm.get_cmap('seismic')

for curKom in allKoms:
        
    curDays,curCount,curPerc = getKommuneCount(curKom)

    firstDate = np.datetime64('2021-12-15')
    # firstDate = np.datetime64('2021-09-01')
    curCount = curCount[curDays >= firstDate]
    curPerc = curPerc[curDays >= firstDate]
    curDays = curDays[curDays >= firstDate]

    curVal = np.cumsum(curPerc)[-1]

    # [cmap(x/7) for x in posVals]
    curColor = cmap(curVal)
    
    if (curKom == 'Aabenraa'):
        curKom = 'Åbenrå'
    if (curKom == 'Nordfyn'):
        curKom = 'Nordfyns'
    if (curKom == 'København'):
        curKom = 'Københavns'
    if (curKom == 'Bornholm'):
        curKom = 'Bornholms'
    if (curKom == 'Faaborg-Midtfyn'):
        curKom = 'Fåborg-Midtfyn'
    if (curKom == 'Lyngby-Taarbæk'):
        curKom = 'Lyngby-Tårbæk'
        
    curgdf.loc[curgdf.navn == (curKom+' Kommune'),'TotalImmunity'] = curVal

    # curgdf = gdf[gdf.navn == (curKom+' Kommune')]
    # curgdf.plot(ax=ax1,color=curColor)

    
# curgdf.head()

# %%


# import matplotlib

fig,ax1 = plt.subplots(figsize=(15,15)) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size="5%", pad=0.05)

fig.patch.set_facecolor('xkcd:off white')
ax1.set_facecolor('xkcd:off white')

# cmap = matplotlib.cm.get_cmap('cividis')
vmax = np.round(curgdf.TotalImmunity.max()/5)*5 + 5
curRangeToShow = np.arange(0,vmax,2.5)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:purple","xkcd:light gray","xkcd:orange"],N=len(curRangeToShow))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:red","xkcd:light gray","b"],N=len(curRangeToShow))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:dark red","k","xkcd:green"],N=len(curRangeToShow))
curgdf.plot(ax=ax1,column = 'TotalImmunity',cmap=cmap,vmax=vmax,legend=True, cax=cax)
curgdf.plot(ax=ax1,column = 'TotalImmunity',cmap=cmap,vmax=vmax,vmin=10,legend=True, cax=cax)


# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

ax1.set_title('Naturlig immunitet fra Omikron smitte, samlet befolkning')
cax.yaxis.set_label_position("left")
cax.yaxis.tick_left()
textFirstDate = pd.to_datetime(firstDate).strftime('%#d. %B %Y')
textLastDate = pd.to_datetime(np.datetime64('today')).strftime('%#d. %B %Y')
cax.set_ylabel(f'Andel af befolkning smittet siden {textFirstDate} [%]')
ax1.set_title(f'Andel af befolkning smittet fra {textFirstDate} til {textLastDate}\nBaseret på PCR-positive resultater, ikke test-justeret')

fig.savefig(path_figs+f'Maps\\Kommune_Total')

# %%
# curgdf.loc[curgdf.navn == (curKom+' Kommune'),'TotalImmunity'] = 10
curgdf.head()

# %%

# curDays,curCount,curPerc = getKommuneCount('Horsens')

# firstDate = np.datetime64('2021-12-01')
# # firstDate = np.datetime64('2021-09-01')
# curCount = curCount[curDays >= firstDate]
# curPerc = curPerc[curDays >= firstDate]
# curDays = curDays[curDays >= firstDate]

# fig,ax1 = plt.subplots()
# ax1.plot(curDays,np.cumsum(curPerc))

# %% [markdown]
# # Make the same map, but age-specific
# 
# Method for getting age-specific case-counts and incidence was made in Kommune_Alder.ipynb

# %%
popdf1 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del1.csv',header=None,encoding='latin1',delimiter=';')
popdf2 = pd.read_csv(rootdir_data+'/DKfolketal2021_Statistikbanken_Del2.csv',header=None,encoding='latin1',delimiter=';')

popdf = pd.concat([popdf1,popdf2])

popdf = popdf.rename(columns={0:"Kommune",1:'Alder',2:'Antal'})
popdf['AlderKort'] = popdf.Alder.apply(lambda x: int(str(x).split(' ')[0]))
totCounts = popdf.groupby('Kommune').sum()

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

df_kommunekort = pd.read_csv(latestdir+'/Kommunalt_DB/10_Kommune_kort.csv',encoding='latin1',
                                delimiter = ';')
df_kommunekort = df_kommunekort.set_index("Kommunenavn")


# %%
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
    

# %%
def getCurrentAgeCounts(curKommuneNavn,Age):
    # ,curMinAge=0,curMaxAge=125
    
    if (Age == '80+'):
        curMinAge,curMaxAge = 80,125
    else:
        curMinAge,curMaxAge = [int(x) for x in Age.split('-')]

    komCode = df_kommunekort['Kommune'][curKommuneNavn]
    curDates,curCounts = getDiffTimeSeries(komCode,Age)

    curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)
    curPerc = curCounts/curPopSize

    return curDates,curCounts,curPerc



# %%
def getCurrentAgeCumulative(curKommuneNavn,Age,firstDate=np.datetime64('2021-12-01')):

    curDates,curCounts,curPerc = getCurrentAgeCounts(curKommuneNavn,Age)
    
    curIndex = [np.datetime64(x) for x in curDates] >= firstDate
    curCounts = curCounts[curIndex]
    curPerc = curPerc[curIndex]
    curDates = curDates[curIndex]

    curVal = np.cumsum(curPerc)[-1]

    return curVal

print(getCurrentAgeCumulative(curKommuneNavn,'0-2'))
print(getCurrentAgeCumulative(curKommuneNavn,'3-5'))
print(getCurrentAgeCumulative(curKommuneNavn,'6-11'))
print(getCurrentAgeCumulative(curKommuneNavn,'40-64'))

# %%
# def getCurrentAgeCumulative(curKommuneNavn,Age,firstDate=np.datetime64('2021-12-01')):
#     # ,curMinAge=0,curMaxAge=125
    
#     if (Age == '80+'):
#         curMinAge,curMaxAge = 80,125
#     else:
#         curMinAge,curMaxAge = [int(x) for x in Age.split('-')]

#     komCode = df_kommunekort['Kommune'][curKommuneNavn]
#     curDates,curCounts = getDiffTimeSeries(komCode,Age)

#     curPopSize = getPopSize(curKommuneNavn,curMinAge,curMaxAge)
#     curPerc = curCounts/curPopSize

#     # firstDate = np.datetime64('2021-12-01')
#     # firstDate = np.datetime64('2021-09-01')
#     curIndex = [np.datetime64(x) for x in curDates] >= firstDate
#     curCounts = curCounts[curIndex]
#     curPerc = curPerc[curIndex]
#     curDates = curDates[curIndex]

#     curVal = np.cumsum(curPerc)[-1]

#     # plt.figure()
#     # plt.plot(curDates,curCounts,'.:')
#     # plt.plot(rnTime(curDates,7),rnMean(curCounts,7)) 

#     return curVal

# print(getCurrentAgeCumulative(curKommuneNavn,'0-2'))
# print(getCurrentAgeCumulative(curKommuneNavn,'3-5'))
# print(getCurrentAgeCumulative(curKommuneNavn,'6-11'))
# print(getCurrentAgeCumulative(curKommuneNavn,'40-64'))

# %%

# fig,ax1 = plt.subplots(figsize=(15,15)) 

# for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf.navn):
#     ax1.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points")

# cmap = matplotlib.cm.get_cmap('seismic')

for curKom in allKoms:
        
    # curDays,curCount,curPerc = getKommuneCount(curKom)

    # firstDate = np.datetime64('2021-12-01')
    # # firstDate = np.datetime64('2021-09-01')
    # curCount = curCount[curDays >= firstDate]
    # curPerc = curPerc[curDays >= firstDate]
    # curDays = curDays[curDays >= firstDate]

    # curVal = np.cumsum(curPerc)[-1]

    # [cmap(x/7) for x in posVals]
    curColor = cmap(curVal)


    curKomGdf = curKom
    if (curKom == 'Aabenraa'):
        curKomGdf = 'Åbenrå'
    if (curKom == 'Nordfyn'):
        curKomGdf = 'Nordfyns'
    if (curKom == 'København'):
        curKomGdf = 'Københavns'
    if (curKom == 'Bornholm'):
        curKomGdf = 'Bornholms'
    if (curKom == 'Faaborg-Midtfyn'):
        curKomGdf = 'Fåborg-Midtfyn'
    if (curKom == 'Lyngby-Taarbæk'):
        curKomGdf = 'Lyngby-Tårbæk'
        
    # curgdf.loc[curgdf.navn == (curKomGdf+' Kommune'),'TotalImmunity'] = curVal
    try: 
        for curAge in df.Aldersgruppe.unique()[:-1]:
            curVal = getCurrentAgeCumulative(curKom,curAge)
            curgdf.loc[curgdf.navn == (curKomGdf+' Kommune'),curAge] = 100 *  curVal
    except:
        print(curKom)

    # curgdf = gdf[gdf.navn == (curKom+' Kommune')]
    # curgdf.plot(ax=ax1,color=curColor)

    
# curgdf.head()

# %%

# curgdf['TotalImmunity'] = curgdf['TotalImmunity']/100
curgdf.head()

# %% [markdown]
# # ... and make maps

# %%
curgdf[curgdf.navn == 'Læsø Kommune']

# %%

for curAge in df.Aldersgruppe.unique()[:-1]:

    fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="5%", pad=0.01)

    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')


    # vmax = np.round(curgdf[curAge].max()/5)*5 + 5
    # curRangeToShow = np.arange(0,vmax,2.5)
    vmax = 60
    vmax = 40
    curRangeToShow = np.arange(0,vmax,5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:dark yellow","gray","xkcd:green"],N=len(curRangeToShow))
    curgdf.plot(ax=ax1,column = curAge,cmap=cmap,legend=True, cax=cax)


    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_title(f'Naturlig immunitet fra Omikron smitte, {curAge} årige')
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    textFirstDate = pd.to_datetime(firstDate).strftime('%#d. %B')
    cax.set_ylabel(f'Andel af {curAge} årige smittet siden {textFirstDate} [%]')

    fig.savefig(path_figs+f'Maps\\Kommune_Immunitet_{curAge}')

# %%

for curAge in df.Aldersgruppe.unique()[:-1]:

    fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="5%", pad=0.01)

    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')


    # vmax = np.round(curgdf[curAge].max()/5)*5 + 5
    # curRangeToShow = np.arange(0,vmax,2.5)
    vmax = 60
    vmax = 40
    curRangeToShow = np.arange(0,vmax,5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:dark yellow","gray","xkcd:green"],N=len(curRangeToShow))
    curgdf.plot(ax=ax1,column = curAge,cmap=cmap,vmax=vmax,vmin=0,legend=True, cax=cax)


    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_title(f'Naturlig immunitet fra Omikron smitte, {curAge} årige')
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    textFirstDate = pd.to_datetime(firstDate).strftime('%#d. %B')
    cax.set_ylabel(f'Andel af {curAge} årige smittet siden {textFirstDate} [%]')

    fig.savefig(path_figs+f'Maps\\Kommune_{curAge}')

# %%
plt.close('all')

# %% [markdown]
# # Make a map of the current age-specific incidence

# %%

# curDates,curCounts,curPerc = getCurrentAgeCounts('København','3-5')
# 100*rnMean(curPerc,7)

# %%

curDays,curCount,curPerc = getKommuneCount('København')
# getKommuneCount('København')
# curPerc

# %%
incigdf = curgdf.copy()
for curKom in allKoms:
        
    curDays,curCount,curPerc = getKommuneCount(curKom)

    firstDate = np.datetime64('2021-12-01')
    # firstDate = np.datetime64('2021-09-01')
    curCount = curCount[curDays >= firstDate]
    curPerc = curPerc[curDays >= firstDate]
    curDays = curDays[curDays >= firstDate]

    # curVal = np.cumsum(curPerc)[-1]
    curVal = rnMean(curPerc,7)[-1]

    # [cmap(x/7) for x in posVals]
    curColor = cmap(curVal)


    curKomGdf = curKom
    if (curKom == 'Aabenraa'):
        curKomGdf = 'Åbenrå'
    if (curKom == 'Nordfyn'):
        curKomGdf = 'Nordfyns'
    if (curKom == 'København'):
        curKomGdf = 'Københavns'
    if (curKom == 'Bornholm'):
        curKomGdf = 'Bornholms'
    if (curKom == 'Faaborg-Midtfyn'):
        curKomGdf = 'Fåborg-Midtfyn'
    if (curKom == 'Lyngby-Taarbæk'):
        curKomGdf = 'Lyngby-Tårbæk'
        
    incigdf.loc[incigdf.navn == (curKomGdf+' Kommune'),'TotalIncidence'] = curVal
    try: 
        for curAge in df.Aldersgruppe.unique()[:-1]:
            # curVal = getCurrentAgeCumulative(curKom,curAge)
            
            curDates,curCounts,curPerc = getCurrentAgeCounts(curKom,curAge)
            curVal = 100*rnMean(curPerc,7)[-1]
            incigdf.loc[incigdf.navn == (curKomGdf+' Kommune'),curAge] = curVal
    except:
        print(curKom)


# %%
incigdf.head()
# curgdf.head()
# # curDays,curCount,curPerc = getKommuneCount('Horsens')

# # plt.figure()
# # plt.plot(rnTime(curDays,7),rnMean(curPerc,7))

# curDates,curCounts,curPerc = getCurrentAgeCounts(curKom,curAge)

# curVal = 100*rnMean(curPerc,7)[-1]
# curVal

# %%
df.Aldersgruppe.unique()[1]
df.Aldersgruppe.unique()[1]
curAge
incigdf.max()

# %%


# %%


fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size="5%", pad=0.01)

fig.patch.set_facecolor('xkcd:off white')
ax1.set_facecolor('xkcd:off white')

# vmax = np.round(curgdf[curAge].max()/5)*5 + 5
# curRangeToShow = np.arange(0,vmax,2.5)
vmax = 1
curRangeToShow = np.arange(0,vmax,0.05)
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:dark red","k","xkcd:green"],N=len(curRangeToShow))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["k","xkcd:blue", "xkcd:red","xkcd:dark red"],N=len(curRangeToShow))
incigdf.plot(ax=ax1,column = 'TotalIncidence',cmap=cmap,vmax=vmax,vmin=0,legend=True, cax=cax)


# Remove axes
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

ax1.set_title(f'Incidens per 100.000 borgere, samlet befolkning')
cax.yaxis.set_label_position("left")
cax.yaxis.tick_left()
textFirstDate = pd.to_datetime(firstDate).strftime('%#d. %B')
cax.set_ylabel(f'Incidens per 100.000 borgere, samlet befolkning')

fig.savefig(path_figs+f'Maps\\Kommune_Incidens_Total')

# %%

for curAge in df.Aldersgruppe.unique()[:-1]:
# for curAge in df.Aldersgruppe.unique()[1:2]:

    fig,ax1 = plt.subplots(figsize=(15,15),tight_layout=True) 

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="5%", pad=0.01)

    fig.patch.set_facecolor('xkcd:off white')
    ax1.set_facecolor('xkcd:off white')

    # vmax = np.round(curgdf[curAge].max()/5)*5 + 5
    # curRangeToShow = np.arange(0,vmax,2.5)
    vmax = 2.5
    curRangeToShow = np.arange(0,vmax,0.05)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["xkcd:dark red","k","xkcd:green"],N=len(curRangeToShow))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["k","xkcd:blue", "xkcd:red","xkcd:dark red"],N=len(curRangeToShow))
    incigdf.plot(ax=ax1,column = curAge,cmap=cmap,vmax=vmax,vmin=0,legend=True, cax=cax)


    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_title(f'Incidens per 100.000 borgere, {curAge} årige')
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    textFirstDate = pd.to_datetime(firstDate).strftime('%#d. %B')
    cax.set_ylabel(f'Incidens per 100.000 borgere, {curAge} årige')

    fig.savefig(path_figs+f'Maps\\Kommune_Incidens_{curAge}')


