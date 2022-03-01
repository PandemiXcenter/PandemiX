# Collection of various functions to use in various PandemiX work, for easy access

# To import, file-directory can be temporarily added to path.
# (Adding permanently to path is also an option, but this is more portable)
# import sys
# sys.path.append("./../scripts")
# import PandemiXFunctions

import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
# ax1.spines['top'].set_visible(False) 

# Define paths
rootdir_data = os.getcwd() +"\\..\\DanskeData\\" 

path_data = rootdir_data + "ssi_data\\"
path_dash = rootdir_data + "ssi_dashboard\\"
path_vacc = rootdir_data + "ssi_vacc\\"

path_figs = os.getcwd() +"\\..\\Figures\\" 

def rnMean(data,meanWidth=7):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTime(t,meanWidth=7):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]
def plotMean(xVals,yVals,ax,meanWidth=7, **kwargs):
    
    line2d_1 = ax.plot(xVals,yVals,'.:',linewidth=0.5,markersize=2, **kwargs)
    line2d_2 = ax.plot(rnTime(xVals,meanWidth),rnMean(yVals,meanWidth),**kwargs)

    return (line2d_1,line2d_2)

def addWeekendsToAx(ax):
    
    # Draw weekends
    firstSunday = np.datetime64('2020-02-02')
    numWeeks = 200
    for k in range(-numWeeks,numWeeks):
        curSunday = firstSunday + np.timedelta64(7*k,'D')
        ax.axvspan(curSunday-np.timedelta64(1,'D')-np.timedelta64(12,'h'),curSunday+np.timedelta64(12,'h'),zorder=-1,facecolor='lightgrey',label=int(k==0)*'Weekend')
    # ax.grid(axis='y')

def getCases(type='raw',includeLatest=False,includeReinfections=True,returnReinfections = False,firstDate=np.datetime64('2020-01-27'),lastDate=np.datetime64('today')):
    # Returns number of positive cases and the correpsponding days
    # With no additional arguments: Raw daily counts, whole data-set, re-infections included in count.
    # # Arguments:
    # type: 
    #     'raw','7daymean','14daymean','21daymean','28daymean','WeeklySum'
    # includeLatest: 
    #     Data is based on testing data, so latest data-point is usually not counted yet. Default: False
    # includeReinfections:
    #     Whether to return the sum of new cases and reinfections or only new cases. Default: True 
    # returnReinfections:
    #     Flag for returning reinfections seperately in case the sum of new cases and reinfection is not returned (see above). Default: False 
    #     Note: If true, 3 arrays are returned: newCases,dates,reinfections. If false, only 2 arrays.
    # firstDate:
    #     Data before day is cutoff (before running means is calculated)
    # lastDate:
    #     Data after day is cutoff (before running means is calculated)


    # # Using data from SSI_data does not include reinfektion. However, the loading is kept here for posterity...
    # latestsubdir = list(os.walk(path_data))[0][1][-1]
    # latestdir = path_data + latestsubdir
    # dfCase = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
    # dfCase = dfCase.iloc[:-2]
    # dfCase['NewPositive'] = pd.to_numeric(dfCase['NewPositive'].astype(str).apply(lambda x: x.replace('.','')))
    # dfCase['Date'] =  pd.to_datetime(dfCase.Date,format='%Y-%m-%d')
    # # Remove dates outside range
    # dfCase = dfCase[(dfCase.Date >= firstDate) & (dfCase.Date <= lastDate)]
    # # Latest datapoint is not yet fully counted and should often be ignored.
    # if (includeLatest):
    #     curCases = dfCase['NewPositive'].values
    #     curDates = dfCase['Date'].values
    # else:
    #     curCases = dfCase['NewPositive'].values[:-1]
    #     curDates = dfCase['Date'].values[:-1]

    # Correct loading, using the regional dashboard files which include both new cases and reinfections
    latestsubdir = list(os.walk(path_dash))[0][1][-1]
    latestdir = path_dash + latestsubdir

    df = pd.read_csv(latestdir+'/Regionalt_DB/24_reinfektioner_daglig_region.csv',encoding='latin1',delimiter = ';')
    df['Prøvedato'] = pd.to_datetime(df['Prøvedato'])
    # groupdf = df.groupby(['Prøvedato').sum()
    df_reinf = df[df['Type af tilfælde (reinfektion eller bekræftet tilfælde)'] == '1.Reinfektion'].groupby('Prøvedato').sum()
    df_inf = df[df['Type af tilfælde (reinfektion eller bekræftet tilfælde)'] != '1.Reinfektion'].groupby('Prøvedato').sum()

    curDates = df_inf.index.values 

    newCases = df_inf.infected.values
    reInfCases = df_reinf.infected.values



    # if includeReinfections:
    #     curCases = newCases + reInfCases
    # else:
    #     curCases = newCases

    # Latest datapoint is not yet fully counted and should often be ignored.
    if (includeLatest == False):
        # curCases = curCases[:-1]
        curDates = curDates[:-1]
        newCases = newCases[:-1]
        reInfCases = reInfCases[:-1]
    
    if (type == '7daymean'):
        curDates = rnTime(curDates,7)
        newCases = rnMean(newCases,7)
        reInfCases = rnMean(reInfCases,7)
    elif (type == '14daymean'):
        curDates = rnTime(curDates,14)
        newCases = rnMean(newCases,14)
        reInfCases = rnMean(reInfCases,14)
    elif (type == '21daymean'):
        curDates = rnTime(curDates,21)
        newCases = rnMean(newCases,21)
        reInfCases = rnMean(reInfCases,21)
    elif (type == '28daymean'):
        curDates = rnTime(curDates,28)
        newCases = rnMean(newCases,28)
        reInfCases = rnMean(reInfCases,28)
    elif (type == 'WeeklySum'):
        # Summing from monday to sunday.
        # If firstDate is not a monday, some days are missing in beginning
        # If lastDate is not a sunday, some days are missing in the end
        
        # Since the dates returned are the mondays of the given week, 
        # the correct way to plot is step with where='post', i.e. ax.step(dates,cases,where='post')

        curWeekDay = pd.to_datetime(firstDate).dayofweek
        firstMonday = firstDate - np.timedelta64(curWeekDay,'D')

        curOffset = 0
        curMonday = firstMonday

        weekNewCases = []
        weekReInfCases = []
        weekCurDates = []
        while (curMonday < lastDate):
            weekNewCases.append(newCases[curOffset:curOffset+7].sum())
            weekReInfCases.append(reInfCases[curOffset:curOffset+7].sum())
            weekCurDates.append(curMonday)

            curMonday = curMonday + np.timedelta64(7,'D')
            curOffset = curOffset + 7
        
        # toReturn = np.array(weekNewCases),np.array(weekCurDates)
        curDates = weekCurDates
        newCases = weekNewCases
        reInfCases = weekReInfCases
    
    if includeReinfections:
        return (newCases+reInfCases),curDates
    else:
        if returnReinfections:
            return newCases,curDates,reInfCases
        else:
            return newCases,curDates
    
def getLatest(type='cases'):

    dfToReturn = pd.DataFrame()

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------- CASES ----------------------------------------
    # ---------------------------------------------------------------------------------------
    # Cases are based on date of test
    if (type == 'cases'):
        # Re-infection data is not available before 2021-01-01, so the old data file is used first:
        latestsubdir = list(os.walk(path_data))[0][1][-1]
        latestdir = path_data + latestsubdir
        dfCase = pd.read_csv(latestdir+'/Test_pos_over_time.csv',delimiter = ';',dtype=str)
        dfCase = dfCase.iloc[:-2]
        dfCase['NewPositive'] = pd.to_numeric(dfCase['NewPositive'].astype(str).apply(lambda x: x.replace('.','')))
        dfCase['Date'] =  pd.to_datetime(dfCase.Date,format='%Y-%m-%d')
        # Remove dates outside range
        dfCase = dfCase[dfCase.Date < np.datetime64('2021-01-01')]

        dfToReturn1 = pd.DataFrame()
        dfToReturn1['Date'] = dfCase.Date
        dfToReturn1['NewCases'] = dfCase.NewPositive 
        dfToReturn1['Reinfections'] = np.nan*dfCase.NewPositive
        dfToReturn1['Total'] = dfCase.NewPositive
        
        # Correct loading, using the regional dashboard files which include both new cases and reinfections
        latestsubdir = list(os.walk(path_dash))[0][1][-1]
        latestdir = path_dash + latestsubdir

        df = pd.read_csv(latestdir+'/Regionalt_DB/24_reinfektioner_daglig_region.csv',encoding='latin1',delimiter = ';')
        df['Prøvedato'] = pd.to_datetime(df['Prøvedato'])
        # groupdf = df.groupby(['Prøvedato').sum()
        df_reinf = df[df['Type af tilfælde (reinfektion eller bekræftet tilfælde)'] == '1.Reinfektion'].groupby('Prøvedato').sum()
        df_inf = df[df['Type af tilfælde (reinfektion eller bekræftet tilfælde)'] != '1.Reinfektion'].groupby('Prøvedato').sum()

        curDates = df_inf.index.values 

        newCases = df_inf.infected.values
        reInfCases = df_reinf.infected.values

        dfToReturn2 = pd.DataFrame()
        dfToReturn2['Date'] = curDates
        dfToReturn2['NewCases'] = newCases 
        dfToReturn2['Reinfections'] = reInfCases
        dfToReturn2['Total'] = newCases+reInfCases

    
        dfToReturn = pd.concat([dfToReturn1,dfToReturn2],ignore_index=True)

    # ---------------------------------------------------------------------------------------
    # ------------------------------------- ADMISSIONS --------------------------------------
    # ---------------------------------------------------------------------------------------
    # Admissions are based on date of registration
    elif (type == 'admissions'):
        # Load data from "noegletal"
        # Until 2021-12-20, all dates were included in one file. Since then, additional data was added, and the file only contains the most recent numbers

        latestsubdirs_dash = list(os.walk(path_dash))[0][1]
        # latestsubdirs_dash == 'SSI_dashboard_2021-12-17'
        lastFullFileIndex = np.where([x == 'SSI_dashboard_2021-12-17' for x in latestsubdirs_dash])[0][0]
        latestdir_dash = path_dash + latestsubdirs_dash[lastFullFileIndex]

        dfKey = pd.read_csv(latestdir_dash+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')

        dfKeysArray = []
        for k in range(lastFullFileIndex+1,len(latestsubdirs_dash)):
            
            latestdir_dash = path_dash + latestsubdirs_dash[k]
            curdf = pd.read_csv(latestdir_dash+'\\Kommunalt_DB\\01_noegletal.csv',encoding='latin1',delimiter=';')
            dfKeysArray.append(curdf)
            

        dfKey['IndberetningDato'] = pd.to_datetime(dfKey['IndberetningDato'])

        # Make arrays to plot
        keyDates = dfKey.IndberetningDato
        keyDatesShift = keyDates + np.timedelta64(365,'D')
        keyCase = dfKey['Antal nye bekræftede tilfælde']
        keyNewAdm = dfKey['Antal nye indlæggelser']
        keyAdm = dfKey['Antal indlagte i dag med COVID']
        keyAdmInt = dfKey['Antal indlagt i dag på intensiv']
        keyAdmResp = dfKey['Antal indlagt i dag og i respirator']
        keyDeath = dfKey['Antal nye døde']

        ## Add the new data

        # 2021-12-20 still used old names
        dateToAdd = np.datetime64(pd.to_datetime(dfKeysArray[0].IndberetningDato.values[0]))
        keyDates = np.append(keyDates,dateToAdd)
        keyCase = np.append(keyCase,dfKeysArray[0]['Antal nye bekræftede tilfælde'][0])
        keyNewAdm = np.append(keyNewAdm,dfKeysArray[0]['Antal nye indlæggelser'][0])
        keyAdm = np.append(keyAdm,dfKeysArray[0]['Antal indlagte i dag med COVID'][0])
        keyAdmInt = np.append(keyAdmInt,dfKeysArray[0]['Antal indlagt i dag på intensiv'][0])
        keyAdmResp = np.append(keyAdmResp,dfKeysArray[0]['Antal indlagt i dag og i respirator'][0])
        keyDeath = np.append(keyDeath,dfKeysArray[0]['Antal nye døde'][0])

        # Make an array for missing reinfection data
        keyCaseReInf = keyCase * np.nan 

        # After which the new names are used
        for k in range(1,len(dfKeysArray)):
            thisDate = dfKeysArray[k].Dato[0]
            thisCase = dfKeysArray[k]['Bekræftede tilfælde siden sidste opdatering'][0]
            thisNewAdm = dfKeysArray[k]['Nyindlæggelser siden sidste opdatering'][0]
            thisDeath = dfKeysArray[k]['Dødsfald siden sidste opdatering'][0]
            thisAdm = dfKeysArray[k]['Indlæggelser i dag'][0]
            thisAdmInt = dfKeysArray[k]['Indlæggelser i dag (intensiv)'][0]
            thisAdmResp = dfKeysArray[k]['Indlæggelser i dag (respirator)'][0]
            
            
            thisCaseReInf = dfKeysArray[k]['Reinfektioner siden sidste opdatering'][0]

            keyDates = np.append(keyDates,np.datetime64(thisDate))
            keyCase = np.append(keyCase,thisCase)
            keyNewAdm = np.append(keyNewAdm,thisNewAdm)
            keyAdm = np.append(keyAdm,thisAdm)
            keyAdmInt = np.append(keyAdmInt,thisAdmInt)
            keyAdmResp = np.append(keyAdmResp,thisAdmResp)
            keyDeath = np.append(keyDeath,thisDeath)

            keyCaseReInf = np.append(keyCaseReInf,thisCaseReInf)


        keyDates = keyDates.astype('datetime64[D]')

        # Collect everything in a single dataframe
        dfToReturn = pd.DataFrame()
        dfToReturn['Date'] = keyDates
        dfToReturn['Cases_New'] = keyCase
        dfToReturn['Cases_Reinfection'] = keyCaseReInf
        dfToReturn['New_Admissions'] = keyNewAdm
        dfToReturn['Hospitalizations'] = keyAdm
        dfToReturn['ICU'] = keyAdmInt
        dfToReturn['Respirator'] = keyAdmResp
        dfToReturn['Deaths'] = keyDeath

    else:
        print('Error, invalid type of data asked for in getLatest-function')

    return dfToReturn

