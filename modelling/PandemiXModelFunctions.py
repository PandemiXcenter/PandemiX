# PandemiXModelFunctions

import numpy as np
from scipy.integrate import solve_ivp

## Various basic systems of differential equations
def SIRmodel(t,x,beta,gamma):

    S,I = x

    dS = - beta * S * I 
    dI =   beta * S * I - gamma * I

    return [dS,dI]
    
# Simple model with two diseases with complete cross-immunity
def SIYRmodel(t,x,beta_I,gamma_I,beta_Y,gamma_Y):

    S,I,Y = x

    dS = - beta_I * S * I - beta_Y * S * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   beta_Y * S * Y - gamma_Y * Y

    return [dS,dI,dY]
    
def getModel(ModelName = 'SIR'):
    if (ModelName == 'SIR'):
        return SIRmodel
    elif (ModelName == 'SIYR'):
        return SIYRmodel
    
## Helper functions
# def simulateModel(ModelFunction=SIRmodel,TimeRange=np.linspace(0,10,100),InitialConditions={'S0':0.99,'I0':0.01},Parameters={'beta': 0.5,'gamma': 1/7}):
def simulateModel(ModelFunction=SIRmodel,TimeRange=np.linspace(0,10,100),InitialConditions=[0.99,0.01],Parameters=[2/7,1/7]):
    
    t0 = TimeRange[0]
    tEnd = TimeRange[-1]
    
    # InitArray = list(InitialConditions.values())
    # ParsArray = list(Parameters.values())
    InitArray =InitialConditions
    ParsArray =Parameters
    
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
        
    
class Scheme:
    def __init__(self,ModelName,InitialConditions,Parameters,tStart,tEnd):
        self.ModelName = ModelName
        self.InitialConditions = InitialConditions
        self.Parameters = Parameters
        self.tStart = tStart
        self.tEnd = tEnd
        
        # Get reference to model
        self.Model = getModel(ModelName)
        
        # Initialize changes
        self.Changes = []
        
    def AddChange(self,ChangeToAdd):
        self.Changes.append(ChangeToAdd)       
        
    
    def simulate(self):
        # If there are no changes, simply simulate
        if (len(self.Changes) == 0):
            tRange = np.linspace(self.tStart,self.tEnd)
            simulationOutput = simulateModel(self.Model,TimeRange=tRange,InitialConditions=self.InitialConditions,Parameters=self.Parameters)
            
            # Add simulation to object
            self.result = type('SimulationResult', (), {})()
            self.result.t = tRange
            self.result.v = simulationOutput
        # If there are changes, simply go through them
        # (Assuming they are in correct order?)
        else:
            tInit = self.tStart
            curInit = self.InitialConditions.copy()
            curPars = self.Parameters.copy()
            
            # For each change, run until the change time
            for i in range(len(self.Changes)):
                curChange = self.Changes[i]
                
                curT = np.linspace(tInit,curChange.t) # TODO: Flag for time-resolution?
                
                print(curInit)
                # Run period
                simulationOutput = simulateModel(self.Model,TimeRange=curT,InitialConditions=curInit,Parameters=curPars)
                
                # Get final values
                finalState = simulationOutput[:,-1]
                # Add variables
                if (len(curChange.AddVariables) > 0):
                    finalState = finalState + curChange.AddVariables
                # Multiply variables
                if (len(curChange.MultiplyVariables) > 0):
                    for j in range(len(finalState)):
                        finalState[j] = finalState[j] * curChange.MultiplyVariables[j]
                        
                # Add to parameters
                if (len(curChange.AddParameters) > 0):
                    curPars = curPars + curChange.AddParameters
                # Multiply Parameters
                if (len(curChange.MultiplyParameters) > 0):
                    for j in range(len(curPars)):
                        curPars[j] = curPars[j] * curChange.MultiplyParameters[j]
                
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
            curT = np.linspace(curChange.t,self.tEnd) # TODO: Flag for time-resolution?
            
            # Run period
            simulationOutput = simulateModel(self.Model,TimeRange=curT,InitialConditions=curInit,Parameters=curPars)
            
            # Append to results
            totT = np.append(totT,curT)                    
            totResult = np.concatenate([totResult,simulationOutput],axis=1)
            
            # Add results to object
            self.result = type('SimulationResult', (), {})()
            self.result.t = totT
            self.result.v = totResult
            
            