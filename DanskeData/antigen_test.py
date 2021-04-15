# Python copy of "antigen_test.ipynb" as of 15/04-2021
# %%
from IPython import get_ipython

# %%
# Notebook til at se på SSI's data for antigentests
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
# Plot different things
firstDate = np.datetime64('2020-03-01') # All tests
firstDate = np.datetime64('2021-02-01') # Antigen test
today = date.today()

# fig,ax1 = plt.subplots(1,1)
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

ax1.plot(df.Date,df.NewPositive,label='Nye positive')
# ax1.plot(df.Date,df.Tested)

ax2.plot(df.Date,100*np.divide(df.NewPositive,df.Tested),label='Positiv procent')
ax2.plot(df.Date,0.1*df.PosPct,label='Positiv procent (Afrundet)')

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax1.set_xlim([firstDate,today])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
plt.tight_layout()

# %% [markdown]
# # Figure to compare antigen positive procent with PCR

# %%
# Plot different things
firstDate = np.datetime64('2020-03-01') # All tests
firstDate = np.datetime64('2021-02-01') # Antigen test
today = date.today()

# fig,ax1 = plt.subplots(1,1)
# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)

# ax1.plot(df.Date,df.NewPositive,label='Nye positive')
ax1.plot(df.Date,df.Tested,label='Tested, antigen')
ax1.plot(dfPCR.Date,dfPCR.Tested,label='Tested, PCR')

ax2.plot(df.Date,df.NewPositive,label='New positive, antigen')
ax2.plot(dfPCR.Date,dfPCR.NewPositive,label='New positive, PCR')

posPctAG = 100*np.divide(df.NewPositive,df.Tested)
posPctPCR = 100*np.divide(dfPCR.NewPositive,dfPCR.Tested)
ax3.plot(df.Date,posPctAG,label='Positiv procent, antigen')
ax3.plot(dfPCR.Date,posPctPCR,label='Positiv procent, PCR')
# ax2.plot(df.Date,df.PosPct,label='Positiv procent (Afrundyet)')
# ax2.plot(dfPCR.Date,0.1*dfPCR.PosPct,label='Positiv procet (Afrundet)')

backToShow = -30

ax1.set_ylim(bottom=0)
ax2Top = np.max(dfPCR.NewPositive[backToShow:])*1.1
ax2.set_ylim(bottom=0,top=ax2Top)
ax3Top =np.max(posPctPCR[backToShow:])*1.1
ax3.set_ylim(bottom=0,top=ax3Top)

ax1.legend()
ax2.legend()
ax3.legend()

ax1.grid()
ax2.grid()
ax3.grid()

# ax1.set_xlim([firstDate,today])
lastDate = df.iloc[-1].Date
ax1.set_xlim([firstDate,lastDate])
# ax1.set_xlim([firstDate,today-np.timedelta64(2,'D')])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))

plt.tight_layout()

if saveFigures:
    fig.savefig('figs/PosPctComparison')


# %%
# df.NewPositive

# %% [markdown]
# # Figures of the "Antigentest_pr_dag.csv" file

# %%
# Load the data
df2 = pd.read_csv(latestdir+'/Antigentests_pr_dag.csv',delimiter = ';')

# # Remove the first two rows which are from april 2020
# df2 = df2.iloc[2:,:].copy()


# %%
df2["Dato"] = df2["Dato"].astype('datetime64[D]')


# %%
# fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)

ax1.plot(df2.Dato,df2.AG_testede,label='AG tests')
ax1.plot(dfPCR.Date,dfPCR.Tested,label='PCR tests')
# ax1.plot(df2.Dato,df2.AG_pos)
# ax1.plot(df2.Dato,100*np.divide(df2.AG_pos,df2.AG_testede),label='Andel af positive AG tests')

# firstToShow = -75
# ax1.plot(df.Date,posPctAG,label='Positiv procent, antigen')
# ax1.plot(dfPCR.Date[firstToShow:],posPctPCR[firstToShow:],label='Positiv procent, PCR')


unconf = df2['AGpos_minusPCRkonf']
unconfAndPos = unconf+df2['AGpos_PCRpos']
allAGPos = unconfAndPos+df2['AGposPCRneg']

# ax2.plot(df2.Dato,df2['AGpos_minusPCRkonf'])
# ax2.plot(df2.Dato,df2['AGpos_PCRpos'])
# ax2.plot(df2.Dato,df2['AGposPCRneg'])

# ax2.plot(df2.Dato,unconf)
# ax2.plot(df2.Dato,unconfAndPos)
# ax2.plot(df2.Dato,allAGPos)
# ax2.plot(df2.Dato,df2.AG_pos)

# ax2.plot(df2.Dato,np.divide(unconf,allAGPos),label='AG positiv, endnu ikke PCR testet')
# ax2.plot(df2.Dato,np.divide(unconfAndPos,allAGPos),label='AG positiv, PCR positiv')
# ax2.plot(df2.Dato,np.divide(allAGPos,allAGPos),label='AG positiv, PCR negativ')

ax2.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),label='AG positiv, PCR negativ')
ax2.fill_between(df2.Dato,100 * np.divide(unconfAndPos,allAGPos),label='AG positiv, PCR positiv')
ax2.fill_between(df2.Dato,100 * np.divide(unconf,allAGPos),color='gray',label='AG positiv, endnu ikke PCR testet')

allConf = df2.AGpos_PCRpos + df2.AGposPCRneg
ax3.plot(df2.Dato,100 * np.divide(df2.AGposPCRneg,allConf),label='AG positiv, PCR negativ')
# ax3.plot(df2.Dato,100 * np.divide(df2.AGpos_PCRpos,allConf),label='AG positiv, PCR positiv')


# sumNeg = np.sum(df2.AGnegPCRpos.values,df2.AGnegPCRneg.values)
sumNeg = df2.AGnegPCRneg.values + df2.AGnegPCRpos.values
# ax3.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),label='AG negativ, PCR positiv')
# ax3.fill_between(df2.Dato,100 * np.divide(df2.AGnegPCRneg,sumNeg),label='AG negativ, PCR negativ')
# ax3.fill_between(df2.Dato,df2.AGnegPCRneg)


# ax3.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),label='AG negativ, PCR negativ')
# ax3.fill_between(df2.Dato,100 * np.divide(df2.AGnegPCRpos,sumNeg),label='AG negativ, PCR positiv')
ax4.plot(df2.Dato,100 * np.divide(df2.AGnegPCRpos,sumNeg),label='AG negativ, PCR positiv')

meanWidth = 7
rmDate = rnTime(df2.Dato,meanWidth)
rmNegPos = rnMean(np.divide(df2.AGnegPCRpos,sumNeg),meanWidth)
ax4.plot(rmDate,100 * rmNegPos,'k:',label='7 dages løbende gennemsnit')


# ax1.legend(loc='upper left')
# ax2.legend(loc='upper left')
ax1.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax2.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax3.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax4.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax1.set_ylim(bottom=0)
ax2.set_ylim([0,100])
ax3.set_ylim([0,100])
ax4.set_ylim([0,100])
# ax4.set_ylim([95,100])
ax4.set_ylim([0,3])

ax1.set_ylabel('Antal tests')
ax2.set_ylabel('Andel af tests [%]')
ax3.set_ylabel('Andel af PCR-\nkonfirmede tests [%]')
ax4.set_ylabel('Andel af tests [%]')
# ax1.set_xlim([df2.iloc[0,0],df2.iloc[-1,0]])

ax1.grid()
# ax2.grid()
ax3.grid()
ax4.grid()

lastDate = df2.iloc[-1].Dato
# ax1.set_xlim([np.datetime64('2021-02-01'),today])
ax1.set_xlim([np.datetime64('2021-02-01'),lastDate])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
plt.tight_layout()

if saveFigures:
    fig.savefig('figs/AGprDagOpsummering')


# %%
# df2


# %%

# fig,ax1 = plt.subplots(1,1)
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

# ax1.plot(df2.Dato,df2['AGnegPCRpos'],label='AG neg, PCR pos')
# ax1.plot(df2.Dato,df2['AGnegPCRneg'],label='AG neg, PCR neg')
# ax1.plot(df2.Dato,1e5*np.divide(df2['AGnegPCRpos'],df2['AG_testede']),label='AG neg, PCR pos')
# ax1.plot(df2.Dato,1e5*np.divide(df2['AGnegPCRneg'],df2['AG_testede']),label='AG neg, PCR neg')

# ax1.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),label='AG positiv, PCR negativ')
# ax1.fill_between(df2.Dato,100 * np.divide(unconfAndPos,allAGPos),label='AG positiv, PCR positiv')
# ax1.fill_between(df2.Dato,100 * np.divide(unconf,allAGPos),color='gray',label='AG positiv, endnu ikke PCR testet')

allConf = df2.AGpos_PCRpos + df2.AGposPCRneg
# ax1.plot(df2.Dato,100 * np.divide(df2.AGpos_PCRpos,allConf),label='AG positiv, PCR positiv')
ax1.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),color='xkcd:beige',label='AG positiv, PCR negativ')
ax1.fill_between(df2.Dato,100 * np.divide(df2.AGpos_PCRpos,allConf),color='xkcd:sky blue',label='AG positiv, PCR positiv')


totAGneg = df2['AGnegPCRpos'] + df2['AGnegPCRneg']
# ax2.plot(df2.Dato,np.divide(df2['AGnegPCRpos'],totAGneg),label='AG neg, PCR pos')
# ax2.plot(df2.Dato,100 * np.divide(df2['AGnegPCRneg'],totAGneg),label='AG negativ, PCR negativ')
ax2.fill_between(df2.Dato,100 * np.ones(df2.Dato.shape),color='xkcd:beige',label='AG negativ, PCR positive')
ax2.fill_between(df2.Dato,100 * np.divide(df2['AGnegPCRneg'],totAGneg),color='xkcd:sky blue',label='AG negativ, PCR negativ')

## Running means
meanWidth = 7
rmDate = rnTime(df2.Dato,meanWidth)
curPP = np.divide(df2.AGpos_PCRpos,allConf)
curNN = np.divide(df2['AGnegPCRneg'],totAGneg)
rmPP = rnMean(curPP,meanWidth)
rmNN = rnMean(curNN,meanWidth)

ax1.plot(rmDate,100 * rmPP,'k:',label=str(meanWidth) + ' dages gennemsnit')
ax2.plot(rmDate,100 * rmNN,'k:',label=str(meanWidth) + ' dages gennemsnit')

firstDateToUse = np.datetime64('2021-02-15')
indexToUse = (df2.Dato > firstDateToUse)
meanPP = np.mean(curPP[indexToUse])
meanNN = np.mean(curNN[indexToUse])
ax1.plot(df2.Dato[indexToUse],100 * np.ones(df2.Dato[indexToUse].shape) * meanPP,'g--',label='Gennemsnit fra '+str(firstDateToUse)+ ' og frem')
ax2.plot(df2.Dato[indexToUse],100 * np.ones(df2.Dato[indexToUse].shape) * meanNN,'g--',label='Gennemsnit fra '+str(firstDateToUse)+ ' og frem')


ax1.set_ylim([0,100])
ax2.set_ylim([95,100])

ax1.legend()
ax2.legend()
# ax1.grid()
# ax2.grid()

lastDate = df2.iloc[-1].Dato
# ax1.set_xlim([np.datetime64('2021-02-01'),today])
ax1.set_xlim([np.datetime64('2021-02-01'),lastDate])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
plt.tight_layout()

if saveFigures:
    fig.savefig('figs/PCRverificeredeAGtest')


# %%
# df2


# %%
# Test of various stuff

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)


# ax1.plot(df.Date,df.Tested)
ax1.plot(df2.Dato,df2['AGnegPCRpos'],'k',label='AG neg, PCR pos')

# ax1.plot(df2.Dato,df2.AG_testede,'k',label='AG neg, PCR pos')

ax2.plot(df2.Dato,100*np.divide(df2['AGnegPCRpos'],df2['AG_testede']),'k',label='AG neg, PCR pos')
# ax1.plot(df2.Dato,df2['AG_testede'],'--')
# ax1.plot(df.Date,df.Tested,':')

# curYlim = ax1.get_ylim()


# sens = 0.99 # TPR
# spec = 0.5 # TNR
# fnr = 1 - sens 
# fpr = 1 - spec
# ax1.plot(df.Date,fnr*df.Tested,label='Forventet antal falske negative')

sensToShow = [0.99,0.995,0.998,0.999]
sensToShow = [0.9975,0.998,0.9985,0.999,0.9995]

for sens in sensToShow:
    fnr = 1 - sens
    # ax1.plot(df.Date,fnr*df.Tested,'--',label='Forventet antal falske negative, sensitivitet: '+str(sens))
    # sensLabel = str(round(100*sens,2))
    sensLabel = "{:2.2f}".format(100*sens)
    # ax1.fill_between(df.Date,fnr*df.Tested,label='Forventet ved sensitivitet: '+sensLabel + ' %')
    ax1.fill_between(df2.Dato,df2['AG_testede']*fnr,label='Forventet ved sensitivitet: '+sensLabel + ' %')

    ax2.fill_between(df.Date,100*fnr*np.ones(df.Date.shape),label='Forventet ved sensitivitet: '+sensLabel+ ' %')

    
# ax2.plot(df2.Dato,100*np.divide(df2['AGnegPCRpos'],df2['AG_testede']),'k',label='__nolegend__')
# ax1.plot(df2.Dato,df2['AGnegPCRpos'],'k',label='__nolegend__')

# ax1.set_ylim([0,curYlim[1]])
ax1.set_ylim(bottom=0)
ax2.set_ylim([0,100*(1-sensToShow[0])])

ax1.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax2.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))

ax1.set_ylabel('Antal falsk negative tests')
ax2.set_ylabel('Andel falsk negative tests [%]')

lastDate = df.iloc[-1].Date
ax1.set_xlim([np.datetime64('2021-02-01'),lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
plt.tight_layout()


if saveFigures:
    fig.savefig('figs/FalskNegativeAG')


# %%
# Test of various stuff

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

ax1.plot(df2.Dato,df2['AGposPCRneg'],'k',label='AG pos, PCR neg')

ax2.plot(df2.Dato,100*np.divide(df2['AGposPCRneg'],df2['AG_testede']),'k',label='AG pos, PCR neg')

# specsToShow = [0.99,0.995,0.998,0.999]
# specsToShow = [0.99,0.995,0.998,0.999]
# specsToShow = [0.9975,0.998,0.9985,0.999,0.9995]
specsToShow = [0.997,0.9975,0.998,0.9985,0.999,0.9995]

for spec in specsToShow:
    fpr = 1 - spec
    specLabel = "{:2.2f}".format(100*spec)
    # ax1.fill_between(df.Date,fpr*df.Tested,label='Forventet ved specificitet: '+specLabel + ' %')
    ax1.fill_between(df2.Dato,df2['AG_testede']*fpr,label='Forventet ved specificitet: '+specLabel + ' %')

    ax2.fill_between(df.Date,100*fpr*np.ones(df.Date.shape),label='Forventet ved specificitet: '+specLabel + ' %')
    
# ax2.plot(df2.Dato,100*np.divide(df2['AGnegPCRpos'],df2['AG_testede']),'k',label='__nolegend__')
# ax1.plot(df2.Dato,df2['AGnegPCRpos'],'k',label='__nolegend__')

# ax1.set_ylim([0,curYlim[1]])
ax1.set_ylim(bottom=0)
ax2.set_ylim([0,100*(1-specsToShow[0])])

ax1.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))
ax2.legend(loc='center left',bbox_to_anchor = (1.0, 0.5))

ax1.set_ylabel('Antal falsk positive tests')
ax2.set_ylabel('Andel falsk positive tests [%]')

lastDate = df.iloc[-1].Date
ax1.set_xlim([np.datetime64('2021-02-01'),lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
plt.tight_layout()
if saveFigures:
    fig.savefig('figs/FalskPositiveAG')


# %%

# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
fig,ax1 = plt.subplots(1,1)

# ax1.plot(df.Date,df.Tested,'.')
# ax1.plot(dfPCR.Date,dfPCR.Tested,'.')

ax1.plot(dfPCR.Date,dfPCR.NewPositive * 1e5 / 5840000,'b.')
beta = 0.55
ax1.plot(dfPCR.Date,dfPCR.NewPositive * ((1e5/dfPCR.Tested)**beta) * 1e5 / 5840000,'r.')


lastDate = df.iloc[-1].Date
ax1.set_xlim([np.datetime64('2021-01-10'),lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
ax1.set_ylim(bottom=0,top = 50)
ax1.grid()
plt.tight_layout()


# %%
firstDate = np.datetime64('2021-02-10')
# lastDate = today
# lastDate = df2.iloc[-1,0]
lastDate = np.datetime64(df2.iloc[-1,0]).astype('datetime64[D]')
curInterval = np.arange(firstDate,lastDate)

indexPCR = dfPCR.Date.isin(curInterval)
indexAG  = df.Date.isin(curInterval)
indexAG2 = df2.Dato.isin(curInterval)

datePCR = dfPCR.Date[indexPCR]
dateAG = df.Date[indexAG]
dateAG2 = df2.Dato[indexAG2]

testPCR = dfPCR.Tested[indexPCR].values
testAG = df.Tested[indexAG].values
testAG2 = df2.AG_testede[indexAG2].values


posPCR = dfPCR.NewPositive[indexPCR].values
posAG = df.NewPositive[indexAG].values
posAG2 = df2.AG_pos[indexAG2].values

posOnlyPCR = posPCR - posAG2
testOnlyPCR = testPCR - posAG2 # Remove those who were only tested because of a positive AG test

beta = 0.55
inciPCR = posPCR * 1e5 / 5_840_000
inciPCRNorm = inciPCR * ((1e5/testPCR)**beta) 
# inciPCRNorm = inciPCR * ((1e5/dfPCR.Tested[indexPCR])**beta) 

inciOnlyPCR = posOnlyPCR * 1e5 / 5_840_000
inciOnlyPCRNorm = inciOnlyPCR * ((1e5/testOnlyPCR)**beta) 


posAGpos = df2.AGpos_PCRpos[indexAG2].values

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)


# ax1.plot(datePCR,5_840_000*inciPCR/1e5,'b.')
# ax1.plot(datePCR,5_840_000*inciPCRNorm/1e5,'r.')
ax1.plot(datePCR,inciPCR,'b.')
ax1.plot(datePCR,inciPCRNorm,'r.')
# ax1.plot(datePCR,inciOnlyPCR,'b*')
# ax1.plot(datePCR,inciOnlyPCRNorm,'r*')

# ax2.plot(dateAG,posAG)
# ax2.plot(dateAG2,posAG2)
# ax2.plot(datePCR,dfPCR.NewPositive[indexPCR])

# ax2.plot(dateAG,np.divide(testAG,inciPCR),'b.')
ax2.plot(dateAG,np.divide(np.multiply(testAG,inciPCR),1e5),'b.')
# ax2.plot(dateAG,np.divide(np.multiply(testAG,inciOnlyPCR),1e5),'b*')
# ax2.plot(dateAG,np.divide(np.multiply(testAG,inciPCRNorm),1e5),'r.')
ax2.plot(dateAG2,posAGpos,'.')


ax1.set_xlim([firstDate,lastDate])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax1.grid()
ax2.grid()
plt.tight_layout()

# plt.figure()
# # plt.plot(df.Date,df.Tested,'.')
# # plt.plot(df2.Dato,df2.AG_testede,'.')
# plt.plot(df2.Dato,100 * np.divide(df2.AG_pos,df2.AG_testede),'.',label='AG positiv procent')
# plt.show()


# %%
# len(testAG)
# len(inciPCR) 

# plt.figure()
# plt.plot(dateAG2,testAG2*(10/1e5),'.')
# plt.show()
# lastDate = np.datetime64(df2.iloc[-1,0]).astype('datetime64[D]')


