import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define paths
rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 

path_data = rootdir_data + "ssi_data\\"
path_dash = rootdir_data + "ssi_dashboard\\"
path_vacc = rootdir_data + "ssi_vacc\\"

path_figs = os.getcwd() +"\\..\\Figures\\" 

def rnMean(data,meanWidth):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTime(t,meanWidth):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]
def getCases(type='raw',includeLatest=False,firstDate=np.datetime64('2020-01-27'),lastDate=np.datetime64('today')):
    latestsubdir = list(os.walk(path_data))[0][1][-1]
    latestdir = path_data + latestsubdir

    dfCase = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
    dfCase = dfCase.iloc[:-2]
    dfCase['NewPositive'] = pd.to_numeric(dfCase['NewPositive'].astype(str).apply(lambda x: x.replace('.','')))
    
    dfCase['Date'] =  pd.to_datetime(dfCase.Date,format='%Y-%m-%d')

    # Remove dates outside range
    dfCase = dfCase[(dfCase.Date >= firstDate) & (dfCase.Date <= lastDate)]
    
    # Latest datapoint is not yet fully counted and should often be ignored.
    if (includeLatest):
        curCases = dfCase['NewPositive'].values
        curDates = dfCase['Date'].values
    else:
        curCases = dfCase['NewPositive'].values[:-1]
        curDates = dfCase['Date'].values[:-1]

    
    if (type == '7daymean'):
        return rnMean(curCases,7),rnTime(curDates,7)
    elif (type == '14daymean'):
        return rnMean(curCases,14),rnTime(curDates,14)
    elif (type == '21daymean'):
        return rnMean(curCases,21),rnTime(curDates,21)
    elif (type == '28daymean'):
        return rnMean(curCases,28),rnTime(curDates,28)
    elif (type == 'WeeklySum'):
        # Summing from monday to sunday.
        # If firstDate is not a monday, some days are missing in beginning
        # If lastDate is not a sunday, some days are missing in the end

        curWeekDay = pd.to_datetime(firstDate).dayofweek
        firstMonday = firstDate - np.timedelta64(curWeekDay,'D')

        curOffset = 0
        curMonday = firstMonday

        casesToReturn = []
        datesToReturn = []
        while (curMonday < lastDate):
            casesToReturn.append(curCases[curOffset:curOffset+7].sum())
            datesToReturn.append(curMonday)

            curMonday = curMonday + np.timedelta64(7,'D')
            curOffset = curOffset + 7
        
        return np.array(casesToReturn),np.array(datesToReturn)
    else:
        return curCases,curDates

def plotMean(xVals,yVals,ax=plt.gca(),meanWidth=7, **kwargs):

    ax.plot(xVals,yVals,'.:',linewidth=0.5,markersize=2, **kwargs)
    ax.plot(rnTime(xVals,meanWidth),rnMean(yVals,meanWidth),**kwargs)

    return 
    
def getLatest():

    return 0
