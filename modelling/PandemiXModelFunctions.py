# PandemiXModelFunctions

import numpy as np
from scipy.integrate import solve_ivp
from copy import copy
import matplotlib.pyplot as plt

## Various basic systems of differential equations
def SIRmodel(t,x,beta,gamma):
    # Standard SIR model.
    # Reduced form: Population (S+I+R) should be 1 
    # 'R' is omitted from equations
    
    S,I = x

    dS = - beta * S * I 
    dI =   beta * S * I - gamma * I

    return [dS,dI]
def SIRmodelMeta():
    return ['S','I'],['beta','gamma']

def SIHRmodel(t,x,beta,gamma,r_chr,u_H):
    # Expansion of SIR model to have hospitalizations as well
    
    # Note that it should not be necessary to actually use this model 
    # since in general hospitalizations, H, can be estimated directly from infections post-simulation:
    # from scipy import integrate
    # int_I = integrate.cumtrapz(np.exp(u_H*ts)*Is,ts, initial=0)
    # Hs = np.exp(-u_H*ts) * (r_chr*pars_full['gamma']*int_I + k) 

    S,I,H = x

    dS = - beta * S * I 
    dI =   beta * S * I - gamma * I
    dH = gamma * r_chr * I  - u_H * H 

    return [dS,dI,dH]
def SIHRmodelMeta():
    return ['S','I','H'],['beta','gamma','r_chr','u_H']
    
# Simple model with two diseases with complete cross-immunity
def SIYRmodel(t,x,beta_I,gamma_I,beta_Y,gamma_Y):

    S,I,Y = x

    dS = - beta_I * S * I - beta_Y * S * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   beta_Y * S * Y - gamma_Y * Y

    return [dS,dI,dY]
def SIYRmodelMeta():
    return ['S','I','Y'],['beta_I','gamma_I','beta_Y','gamma_Y']
    
    
# Uvaccinerede smittede med Omikron: Kun immunitet mod Omikron
# Vaccinerede smittede med Omikron: Immunitet mod Omikron og andre 
# @Tuliodna på twitter


# Two diseases with one-way cross-immunity and one-variant vaccination
def SVIYRRmodel(t,x,beta_I,gamma_I,beta_YS,beta_YV,beta_YR,gamma_Y):
    
    # R_Y is 1-all

    S,V,I,Y,R_I = x

    dS = - beta_I * S * I - beta_YS * S * Y 
    dV = - beta_YV * V * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   (beta_YS * S + beta_YV * V + beta_YR * R_I) * Y - gamma_Y * Y
    dRI =  gamma_I * I - beta_YR * R_I * Y 

    return [dS,dV,dI,dY,dRI]
def SVIYRRmodelMeta():
    return ['S','V','I','Y','R_I'],['beta_I','gamma_I','beta_YS','beta_YV','beta_YR','gamma_Y']
    
def getModel(ModelName = 'SIR'):
    if (ModelName == 'SIR'):
        return SIRmodel,SIRmodelMeta()
    elif (ModelName == 'SIHR'):
        return SIHRmodel,SIHRmodelMeta()
    elif (ModelName == 'SIYR'):
        return SIYRmodel,SIYRmodelMeta()
    elif (ModelName == 'SVIYRR'):
        return SVIYRRmodel,SVIYRRmodelMeta()
        
def DictToArray(curDict,dictMeta):
    curArray =[]
    for i in range(len(curDict)):
        curArray.append(curDict[dictMeta[i]]) 
    return curArray
    
## Helper functions
# def simulateModel(ModelFunction=SIRmodel,TimeRange=np.linspace(0,10,100),InitialConditions={'S0':0.99,'I0':0.01},Parameters={'beta': 0.5,'gamma': 1/7}):
# def simulateModel(ModelFunction=SIRmodel,TimeRange=np.linspace(0,10,100),InitialConditions=[0.99,0.01],Parameters=[2/7,1/7]):
def simulateModel(ModelName='SIR',TimeRange=np.linspace(0,10,100),InitialConditions={'S0':0.99,'I0':0.01},Parameters={'beta': 0.5,'gamma': 1/7}):
    
    ModelFunction,ModelMeta = getModel(ModelName)
    VarsMeta,ParsMeta = ModelMeta
    
    t0 = TimeRange[0]
    tEnd = TimeRange[-1]
    
    # If initial conditions are given as a dictionary, make a list in the correct order
    if (type(InitialConditions) == dict):
        # InitArray =[]
        # for i in range(len(InitialConditions)):
        #     InitArray.append(InitialConditions[VarsMeta[i]]) 
        InitArray = DictToArray(InitialConditions,VarsMeta)
    else: 
        InitArray = InitialConditions
        
    # If parameters are given as a dictionary, make a list in the correct order
    if (type(Parameters) == dict):
        # ParsArray =[]
        # for i in range(len(Parameters)):
        #     ParsArray.append(Parameters[ParsMeta[i]]) 
        ParsArray = DictToArray(Parameters,ParsMeta)
    else: 
        ParsArray = Parameters
        
    # InitArray = list(InitialConditions.values())
    # ParsArray = list(Parameters.values())
    # InitArray =InitialConditions
    # ParsArray =Parameters
    
    
    sol = solve_ivp(ModelFunction,[t0,tEnd],InitArray,t_eval=TimeRange,args=ParsArray)
    
    return sol.y
    
def prettyDict(curDict):
    toReturn = ''
    for key in curDict:
        toReturn += f'{key}: {curDict[key]}, '
    return toReturn[:-2]
    
## Simulation schemes
class Change:
    def __init__(self,tChange,AddVariables = [],MultiplyVariables = [],AddParameters = [],MultiplyParameters = []):
        self.t = tChange 
        self.AddVariables = AddVariables
        self.MultiplyVariables = MultiplyVariables
        self.AddParameters = AddParameters
        self.MultiplyParameters = MultiplyParameters
        
    def __str__(self):
        return self.getStringDescription()
    
    def getStringDescription(self):
        curStr = ''
        curStr += f'At time {self.t}, '
        if (len(self.AddVariables) > 0):
            if type(self.AddVariables) == dict:
                curStr += f'add {prettyDict(self.AddVariables)} to variables, '
            else:
                curStr += f'add {self.AddVariables} to variables, '
        if (len(self.MultiplyVariables) > 0):
            if type(self.MultiplyVariables) == dict:
                curStr += f'multiply variables {prettyDict(self.MultiplyVariables)}, '
            else:
                curStr += f'multiply variables {self.MultiplyVariables}, '
        if (len(self.AddParameters) > 0):
            if type(self.AddParameters) == dict:
                curStr += f'add {prettyDict(self.AddParameters)} to parameters, '
            else:
                curStr += f'add {self.AddParameters} to parameters, '
        if (len(self.MultiplyParameters) > 0):
            if type(self.MultiplyParameters) == dict:
                curStr += f'multiply parameters {prettyDict(self.MultiplyParameters)}, '
            else:
                curStr += f'multiply parameters {self.MultiplyParameters}, '
        # Remove excess comma and space
        curStr = curStr[:-2]
        return curStr 
    
    def __copy__(self):
        return type(self)(self.t,self.AddVariables,self.MultiplyVariables,self.AddParameters,self.MultiplyParameters)
        
    def copy(self):
        return copy(self)
        
    
class Scheme:
    def __init__(self,ModelName,InitialConditions,Parameters,tStart,tEnd,Changes=[]):
        self.ModelName = ModelName
        self.InitialConditions = InitialConditions
        self.Parameters = Parameters
        self.tStart = tStart
        self.tEnd = tEnd
        
        # # Get reference to model
        # self.Model,self.ModelMeta = getModel(ModelName)
        
        # Initialize changes
        self.Changes = Changes
        
    def __str__(self):
        curStr = '-------'
        curStr += f'\nModel: {self.ModelName}.'
        curStr += f'\nComplete simulation running from t={self.tStart} until t={self.tEnd}'
        curStr += f'\nInitial conditions: {self.InitialConditions}'
        curStr += f'\nParameters: {self.Parameters}'
        if (len(self.Changes) > 0):
            curStr += '\n---'
            curStr += '\nChanges: '
            for i in range(len(self.Changes)):
                curCha = self.Changes[i]
                curStr += f'\nChange {i}: {self.Changes[i].getStringDescription()}'
                # curStr += f'\nAt time {curCha.t}...'
                
            curStr += '\n---'
        return curStr
        
    def __copy__(self):
        return type(self)(self.ModelName,self.InitialConditions,self.Parameters,self.tStart,self.tEnd,copy(self.Changes))
    
    def copy(self):
        return copy(self)
        
    def addChange(self,ChangeToAdd):
        self.Changes.append(ChangeToAdd)       
    
    
    def sortChanges(self):
        allTimes = []
        newChanges = []
        for curChange in self.Changes:
            allTimes.append(curChange.t)
        # First check if there are changes with same time
        if (len(allTimes) > len(np.unique(allTimes))):
            print('Some time-points are repeated. Please fix Changes.')
        else:
            newOrderIndex = np.argsort(allTimes)
            
            for i in range(len(newOrderIndex)):
                newChanges.append(self.Changes[newOrderIndex[i]])
                
            self.Changes = newChanges
    
    def simulate(self,tRes=100):
        # If there are no changes, simply simulate
        if (len(self.Changes) == 0):
            tRange = np.linspace(self.tStart,self.tEnd,tRes)
            simulationOutput = simulateModel(self.ModelName,TimeRange=tRange,InitialConditions=self.InitialConditions,Parameters=self.Parameters)
            
            # Add simulation to object
            self.result = type('SimulationResult', (), {})()
            self.result.t = tRange
            self.result.y = simulationOutput
        # If there are changes, simply go through them
        # (Assuming they are in correct order. Changes should be ordered using sortChanges())
        else:
            tInit = self.tStart
            curInit = self.InitialConditions.copy()
            curPars = self.Parameters.copy()
            _,ModelMeta = getModel(self.ModelName)
            VarsMeta,ParsMeta = ModelMeta
            
            # For each change, run until the change time
            for i in range(len(self.Changes)):
                curChange = self.Changes[i]
                
                curT = np.linspace(tInit,curChange.t,tRes) # TODO: Flag for time-resolution?. DONE: Flag is tRes
                
                # Run period
                simulationOutput = simulateModel(self.ModelName,TimeRange=curT,InitialConditions=curInit,Parameters=curPars)
                
                # Get final values
                finalState = simulationOutput[:,-1]
                # Add variables
                # if (type(curChange.AddVariables) == dict):
                #     finalState = finalState + DictToArray(curChange.AddVariables,VarsMeta)
                if (len(curChange.AddVariables) > 0):
                    if (type(curChange.AddVariables) == dict):
                        finalState = finalState + DictToArray(curChange.AddVariables,VarsMeta) 
                    else:
                        finalState = finalState + curChange.AddVariables
                # Multiply variables
                if (len(curChange.MultiplyVariables) > 0):
                    if (type(curChange.MultiplyVariables) == dict):
                        toMult = DictToArray(curChange.MultiplyVariables,VarsMeta) 
                    else:
                        toMult = curChange.MultiplyVariables
                    for j in range(len(finalState)):
                        finalState[j] = finalState[j] * toMult[j]
                        
                # Add to parameters
                if (len(curChange.AddParameters) > 0):
                    if (type(curChange.AddParameters) == dict):
                        curPars = curPars + DictToArray(curChange.AddParameters,ParsMeta) 
                    else:
                        curPars = curPars + curChange.AddParameters
                # Multiply Parameters
                if (len(curChange.MultiplyParameters) > 0):
                    if (type(curChange.MultiplyParameters) == dict):
                        toMult = DictToArray(curChange.MultiplyParameters,ParsMeta) 
                    else:
                        toMult = curChange.MultiplyParameters
                    
                    if (type(curPars) == dict):
                        for key in curPars:
                            curPars[key] = curPars[key] * curChange.MultiplyParameters[key]
                    else:
                        for j in range(len(curPars)):
                            curPars[j] = curPars[j] * toMult[j]
                
                # Set curInit for next part
                curInit = finalState
                
                # If first part, save results
                if (i==0):
                    totT = curT
                    totResult = simulationOutput
                # Otherwise, extend current results
                else:
                    totT = np.append(totT,curT)                    
                    totResult = np.concatenate([totResult,simulationOutput],axis=1)
                    
                # Prepare time for next part
                tInit = curT[-1]
                    
                
            # Run the final period
            curT = np.linspace(curChange.t,self.tEnd,tRes) # TODO: Flag for time-resolution?
            
            # Run period
            simulationOutput = simulateModel(self.ModelName,TimeRange=curT,InitialConditions=curInit,Parameters=curPars)
            
            # Append to results
            totT = np.append(totT,curT)                    
            totResult = np.concatenate([totResult,simulationOutput],axis=1)
            
            # Add results to object
            self.result = type('SimulationResult', (), {})()
            self.result.t = totT
            self.result.y = totResult
            
    def plot(self,fig=[],linestyle='-',color='k',showChanges=True,describeChanges=True):
        
        # If simulation has not been run, run it now
        if (~hasattr(self,'result')):
            self.simulate()
        
        # Get model meta information
        _,ModelMeta = getModel(self.ModelName)
        VarsMeta,ParsMeta = ModelMeta
        
        numPlots = len(VarsMeta)
        # If no figure to plot on is supplied:
        if (type(fig) == list):
            fig,allAxes = plt.subplots(numPlots,1,sharex=True)
        else:
            # If the number of axes already matched the number of variables, it is assumed that the user wants to plot on top of a previous figure
            if len(fig.get_axes())==numPlots:
                allAxes = fig.get_axes()
                # allAxes = fig.get_axes().flatten() # (Flattened, in case different layout was used previously)
            # Otherwise, new subplots are made
            else:
                allAxes = []
                ax1 = fig.add_subplot(numPlots,1,1)
                allAxes.append(ax1)
                for i in range(2,numPlots+1):
                    curAx = fig.add_subplot(numPlots,1,i,sharex=ax1)
                    allAxes.append(curAx)
        
        # Plot results
        ts = self.result.t
        
        for i in range(numPlots):
            curAx = allAxes[i]
            curVarName = VarsMeta[i]
            curAx.plot(ts,self.result.y[i,:],linestyle=linestyle,color=color,label=curVarName)
            
            curAx.set_ylabel(curVarName,fontsize=16)
            
        fig.tight_layout()
        if (showChanges):
            # Add changes to figure
            if len(self.Changes) > 0:
                strFootnote = 'Timed changes:\n'
                for i in range(len(self.Changes)):
                    curChange = self.Changes[i]
                    curT = curChange.t 
                    for j in range(numPlots):
                        curAx = allAxes[j]
                        curAx.axvline(curT,color=color,linestyle=':')
                        
                    strFootnote += curChange.getStringDescription()+'\n'
                
                if describeChanges:
                    t = fig.text(0.02, 0, strFootnote, fontsize=14, ha='left',va='bottom',transform=fig.transFigure)
                    
                    # fig.tight_layout()
                    bb = t.get_window_extent(renderer=fig.canvas.get_renderer()) 
                    figHeight = (fig.get_size_inches()*fig.dpi)[1]
                    fig.subplots_adjust(bottom=0.05+bb.height/figHeight)
                
                # fig.subplots_adjust(bottom=0.2)
                # plt.subplots_adjust(bottom=0.2)
                # plt.figtext(0.05,0.01,strFootnote,ha='left',va='bottom',fontsize=16)
            # fig.tight_layout()
        
        return fig,allAxes