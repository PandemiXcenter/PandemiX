# PandemiXModelFunctions

import numpy as np
from scipy.integrate import solve_ivp
from copy import copy

## Various basic systems of differential equations
def SIRmodel(t,x,beta,gamma):

    S,I = x

    dS = - beta * S * I 
    dI =   beta * S * I - gamma * I

    return [dS,dI]
def SIRmodelMeta():
    return ['S','I'],['beta','gamma']
    
# Simple model with two diseases with complete cross-immunity
def SIYRmodel(t,x,beta_I,gamma_I,beta_Y,gamma_Y):

    S,I,Y = x

    dS = - beta_I * S * I - beta_Y * S * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   beta_Y * S * Y - gamma_Y * Y

    return [dS,dI,dY]
def SIYRmodelMeta():
    return ['S','I','Y'],['beta_I','gamma_I','beta_Y','gamma_Y']
    
def getModel(ModelName = 'SIR'):
    if (ModelName == 'SIR'):
        return SIRmodel,SIRmodelMeta()
    elif (ModelName == 'SIYR'):
        return SIYRmodel,SIYRmodelMeta()
        
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
    InitMeta,ParsMeta = ModelMeta
    
    t0 = TimeRange[0]
    tEnd = TimeRange[-1]
    
    # If initial conditions are given as a dictionary, make a list in the correct order
    if (type(InitialConditions) == dict):
        # InitArray =[]
        # for i in range(len(InitialConditions)):
        #     InitArray.append(InitialConditions[InitMeta[i]]) 
        InitArray = DictToArray(InitialConditions,InitMeta)
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
    ## Simulation schemes
class Change:
    def __init__(self,tChange,AddVariables = [],MultiplyVariables = [],AddParameters = [],MultiplyParameters = []):
        self.t = tChange 
        self.AddVariables = AddVariables
        self.MultiplyVariables = MultiplyVariables
        self.AddParameters = AddParameters
        self.MultiplyParameters = MultiplyParameters
        
    def __str__(self):
        return getStringDescription(self)
    
    def getStringDescription(self):
        curStr = ''
        curStr += f'At time {self.t}, '
        if (len(self.AddVariables) > 0):
            curStr += f'add {self.AddVariables} to variables, '
        if (len(self.MultiplyVariables) > 0):
            curStr += f'multiply variables by {self.MultiplyVariables}, '
        if (len(self.AddParameters) > 0):
            curStr += f'add {self.AddParameters} to parameters, '
        if (len(self.MultiplyParameters) > 0):
            curStr += f'multiply parameters by {self.MultiplyParameters}, '
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
        # (Assuming they are in correct order?)
        else:
            tInit = self.tStart
            curInit = self.InitialConditions.copy()
            curPars = self.Parameters.copy()
            _,ModelMeta = getModel(self.ModelName)
            InitMeta,ParsMeta = ModelMeta
            
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
                #     finalState = finalState + DictToArray(curChange.AddVariables,InitMeta)
                if (len(curChange.AddVariables) > 0):
                    if (type(curChange.AddVariables) == dict):
                        finalState = finalState + DictToArray(curChange.AddVariables,InitMeta) 
                    else:
                        finalState = finalState + curChange.AddVariables
                # Multiply variables
                if (len(curChange.MultiplyVariables) > 0):
                    if (type(curChange.MultiplyVariables) == dict):
                        toMult = DictToArray(curChange.MultiplyVariables,InitMeta) 
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
            
            