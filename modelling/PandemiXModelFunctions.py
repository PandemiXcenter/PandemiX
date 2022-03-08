# PandemiXModelFunctions

import numpy as np
from scipy.integrate import solve_ivp
from copy import copy
import matplotlib.pyplot as plt
       
def DictToArray(curDict,dictMeta,DefaultValues):
    curArray =[]
    # If all variables/parameters are mentioned in dict, 
    # length should be the same, so assume order is correct
    if (len(curDict) == len(dictMeta)):
        for i in range(len(curDict)):
            curArray.append(curDict[dictMeta[i]]) 
    else:
        # Go through each name
        for i in range(len(dictMeta)):
            curStr = dictMeta[i]
            # If the name is in the dict, append the values
            if curStr in curDict:
                curArray.append(curDict[curStr])
            # Otherwise, append zero
            else:
                if (type(DefaultValues) == dict):
                    curArray.append(DefaultValues[curStr])
                else:
                    curArray.append(DefaultValues[i])
    return curArray
    
## Various basic systems of differential equations
def SIRModel(t,x,beta,gamma):
    # Standard SIR model.
    # Reduced form: Population (S+I+R) should be 1 
    # 'R' is omitted from equations
    
    S,I = x

    dS = - beta * S * I 
    dI =   beta * S * I - gamma * I

    return [dS,dI]
def SIRModelMeta():
    return ['S','I'],['beta','gamma']

def SIHRModel(t,x,beta,gamma,r_chr,u_H):
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
def SIHRModelMeta():
    return ['S','I','H'],['beta','gamma','r_chr','u_H']
    
# Simple model with two diseases with complete cross-immunity
def SIYRModel(t,x,beta_I,gamma_I,beta_Y,gamma_Y):

    S,I,Y = x

    dS = - beta_I * S * I - beta_Y * S * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   beta_Y * S * Y - gamma_Y * Y

    return [dS,dI,dY]
def SIYRModelMeta():
    return ['S','I','Y'],['beta_I','gamma_I','beta_Y','gamma_Y']
    
# Two diseases with one-way cross-immunity and one-variant vaccination
def SVIYRRModel(t,x,beta_I,gamma_I,beta_YS,beta_YV,beta_YR,gamma_Y):
    
    # R_Y is 1-all

    S,V,I,Y,R_I = x

    dS = - beta_I * S * I - beta_YS * S * Y 
    dV = - beta_YV * V * Y 
    dI =   beta_I * S * I - gamma_I * I
    dY =   (beta_YS * S + beta_YV * V + beta_YR * R_I) * Y - gamma_Y * Y
    dRI =  gamma_I * I - beta_YR * R_I * Y 

    return [dS,dV,dI,dY,dRI]
def SVIYRRModelMeta():
    return ['S','V','I','Y','R_I'],['beta_I','gamma_I','beta_YS','beta_YV','beta_YR','gamma_Y']
   
def OmikronDeltaFullModel(t,x,beta_IS_S,beta_IV_S,beta_I01_S,beta_IS_V,beta_IV_V,beta_I01_V,beta_IS_R01,beta_IV_R01,beta_I01_R01,beta_Y_S,beta_Y10_S,beta_Y_R10,beta_Y10_R10,gamma_IS,gamma_IV,gamma_Y,gamma_I01,gamma_Y10):
    # Full model describing a suggested behaviour for Omikron and Delta development.
    # Based on the idea that Omikron can infect both vaccinated and unvaccinated, 
    # but Delta (or similar variant) can infect omikron-recovered and unvaccinated. 
    # I.e. "Unvaccinated -> Omikron -> Recovered" only provides protection against Omikron, not delta
    
    # Uvaccinerede smittede med Omikron: Kun immunitet mod Omikron
    # Vaccinerede smittede med Omikron: Immunitet mod Omikron og andre 
    # @Tuliodna på twitter
    
    # Variables: 
    # S: Susceptible (Unvaccinated)
    # V: Vaccinated
    # IS:  Omikron-infected, unvaccinated
    # IV:  Omikron-infected, vaccinated
    # I01: Omikron-infected, Delta-recovered
    # Y:   Delta-infected, (unvaccinated)
    # Y10: Delta-infected, Omikron-recovered
    # R01: Delta-recovered
    # R10: Omikron-recovered
    # R11: Both-recovered
    
    # Parameters:
    # beta_x_y: Rate of transmission from group x to group y
    # gamma_y: Rate of recovery from group y
    # (While beta_x_y1 would probably be similar to beta_x_y2, this implementation allows for different rates)
    # beta_IS_S,beta_IV_S,beta_I01_S,beta_IS_V,beta_IV_V,beta_I01_V,beta_IS_R01,beta_IV_R01,beta_I01_R01,beta_Y_S,beta_Y10_S,beta_Y_R10,beta_Y10_R10,gamma_IS,gamma_IV,gamma_Y,gamma_I01,gamma_Y10 = P
    
    S,V,IS,IV,Y,R01,R10,I01,Y10 = x 
    
    dS   = - (beta_IS_S * IS + beta_IV_S * IV + beta_I01_S * I01 + beta_Y_S * Y + beta_Y10_S * Y10) * S
    dV   = - (beta_IS_V * IS + beta_IV_V * IV + beta_I01_V * I01) * V
    dIS  =   (beta_IS_S * IS + beta_IV_S * IV + beta_I01_S * I01) * S               - gamma_IS * IS 
    dIV  =   (beta_IS_V * IS + beta_IV_V * IV + beta_I01_V * I01) * V               - gamma_IV * IV
    dY   =   (beta_Y_S * Y + beta_Y10_S * Y10) * S                                  - gamma_Y * Y 
    dR01 = - (beta_IS_R01 * IS + beta_IV_R01 * IV + beta_I01_R01 * I01) * R01       + gamma_Y  * Y  
    dR10 = - (beta_Y_R10 * Y + beta_Y10_R10 * Y10) * R10                            + gamma_IS * IS
    dI01 =   (beta_IS_R01 * IS + beta_IV_R01 * IV + beta_I01_R01 * I01) * R01       - gamma_I01 * I01 
    dY10 =   (beta_Y_R10 * Y + beta_Y10_R10 * Y10) * R10                            - gamma_Y10 * Y10
    # dR11 = gamma_IV * IV + gamma_I01 * I01 + gamma_Y10 * Y10
    
    return [dS,dV,dIS,dIV,dY,dR01,dR10,dI01,dY10]   
    
def OmikronDeltaFullModelMeta():
    
    varsMeta = ['S','V','IS','IV','Y','R01','R10','I01','Y10']
    parsMeta = ['beta_IS_S','beta_IV_S','beta_I01_S','beta_IS_V','beta_IV_V','beta_I01_V','beta_IS_R01','beta_IV_R01','beta_I01_R01','beta_Y_S','beta_Y10_S','beta_Y_R10','beta_Y10_R10','gamma_IS','gamma_IV','gamma_Y','gamma_I01','gamma_Y10']

    return varsMeta,parsMeta
 
def getModel(ModelName = 'SIR'):
    # if (ModelName == 'SIR'):
    #     return SIRModel,SIRModelMeta()
    # elif (ModelName == 'SIHR'):
    #     return SIHRModel,SIHRModelMeta()
    # elif (ModelName == 'SIYR'):
    #     return SIYRModel,SIYRModelMeta()
    # elif (ModelName == 'SVIYRR'):
    #     return SVIYRRModel,SVIYRRModelMeta()
    # elif (ModelName == 'OmikronDeltaFull'):
    #     return OmikronDeltaFullModel,OmikronDeltaFullModelMeta()
    
    if (type(ModelName) == str):
        if (ModelName == 'SIR'):
            return SIRModel,SIRModelMeta()
        elif (ModelName == 'SIHR'):
            return SIHRModel,SIHRModelMeta()
        elif (ModelName == 'SIYR'):
            return SIYRModel,SIYRModelMeta()
        elif (ModelName == 'SVIYRR'):
            return SVIYRRModel,SVIYRRModelMeta()
        elif (ModelName == 'OmikronDeltaFull'):
            return OmikronDeltaFullModel,OmikronDeltaFullModelMeta()
    else: 
        # ModelFunction = lambda t,x,*args: ArbitraryFunctionCall(t,x,*args,*ModelName)
        ModelFlow,VarsMeta,ParsMeta = ModelName 
        # ModelFunction = lambda t,x,*args: ModelCallFromMeta(t,x,*args,ModelMeta=ModelFlow)
        ModelFunction = lambda t,x,*args: ArbitraryFunctionCall(t,x,args,ModelFlow,VarsMeta,ParsMeta)

        return ModelFunction,(VarsMeta,ParsMeta)
         
   
    # if (type(ModelFlowAndMeta) == str):
    #     ModelFunction,ModelMeta = getModel(ModelFlowAndMeta)
    #     VarsMeta,ParsMeta = ModelMeta
    # else: 
    #     ModelFunction = lambda t,x,*args: ArbitraryFunctionCall(t,x,*args,*ModelFlowAndMeta)
    #     # ModelFlow,VarsMeta,ParsMeta = ModelFlowAndMeta
    #     # ModelFunction = lambda t,x,*args: ModelCallFromMeta(t,x,*args,ModelMeta=ModelFlow)
         
def getAvailableModels():
    return ['SIR','SIHR','SIYR','SVIYRR','OmikronDeltaFull']

## Arbitrary function definitions
def ArbitraryFunctionCall(t,x,ps,modelFlows,varsMeta,parsMeta):

    # Go through variable names and get inputs
    for i in range(len(varsMeta)):
        vName = varsMeta[i]
        exec(vName+' = x['+str(i)+']')
        
    # Go through parameter names and save
    for i in range(len(parsMeta)):
        pName = parsMeta[i]
        exec(pName+' = ps['+str(i)+']')

    # Calculate the value of each current flow
    allFlows = np.zeros(len(modelFlows))
    i = 0
    for key in modelFlows:
        allFlows[i] = eval(key)
        i = i + 1
        
    # Initialize outputs as zero
    dxdt = np.zeros(x.shape)
    # Go through each variable
    for i in range(len(varsMeta)):
        # # Make sure the current output is zero
        # dxdt[i] = 0
        # Get the name of the current variable
        curVarName = varsMeta[i]
        # Go through each flow
        for j in range(len(modelFlows)):
            # Get the current flow
            curFlow = list(modelFlows.items())[j]
            # Identify inputs and outputs
            curOut = curFlow[1][0]
            curIn = curFlow[1][1]
            # If the flow is an output of the current variable, subtract the flow-value
            if (curOut == curVarName):
                dxdt[i] -= allFlows[j]
            # If the flow is an input of the current variable, add the flow-value
            if (curIn == curVarName):
                dxdt[i] += allFlows[j]

    return dxdt

# # Make a shorter function call for use in solve_ivp
# def ModelCallFromMeta(t,x,*args,ModelMeta):
#     modelFlows,varsMeta,parsMeta = ModelMeta
#     return ArbitraryFunctionCall(t,x,args,modelFlows,varsMeta,parsMeta)

## Helper functions
# def simulateModel(ModelFunction=SIRModel,TimeRange=np.linspace(0,10,100),InitialConditions={'S0':0.99,'I0':0.01},Parameters={'beta': 0.5,'gamma': 1/7}):
# def simulateModel(ModelFunction=SIRModel,TimeRange=np.linspace(0,10,100),InitialConditions=[0.99,0.01],Parameters=[2/7,1/7]):
# def simulateModel(ModelName='SIR',TimeRange=np.linspace(0,10,100),InitialConditions={'S0':0.99,'I0':0.01},Parameters={'beta': 0.5,'gamma': 1/7}):
# def simulateModel(ModelName,TimeRange,InitialConditions,Parameters):
def simulateModel(ModelFlowAndMeta,TimeRange,InitialConditions,Parameters):
    
    ModelFunction,ModelMeta = getModel(ModelFlowAndMeta)
    VarsMeta,ParsMeta = ModelMeta

    # if (type(ModelFlowAndMeta) == str):
    #     ModelFunction,ModelMeta = getModel(ModelFlowAndMeta)
    #     VarsMeta,ParsMeta = ModelMeta
    # else: 
    #     ModelFunction = lambda t,x,*args: ArbitraryFunctionCall(t,x,*args,*ModelFlowAndMeta)
    #     # ModelFlow,VarsMeta,ParsMeta = ModelFlowAndMeta
    #     # ModelFunction = lambda t,x,*args: ModelCallFromMeta(t,x,*args,ModelMeta=ModelFlow)
    
    t0 = TimeRange[0]
    tEnd = TimeRange[-1]
    
    # If initial conditions are given as a dictionary, make a list in the correct order
    if (type(InitialConditions) == dict):
        # InitArray =[]
        # for i in range(len(InitialConditions)):
        #     InitArray.append(InitialConditions[VarsMeta[i]]) 
        # InitArray = DictToArray(InitialConditions,VarsMeta)
        
        # If any values are missing, they will be set to 0
        InitArray = DictToArray(InitialConditions,VarsMeta,np.zeros(len(VarsMeta)))
        # InitArray = DictToArray(InitialConditions,VarsMeta,np.ones(len(VarsMeta)))
    else: 
        InitArray = InitialConditions
        
    # If parameters are given as a dictionary, make a list in the correct order
    if (type(Parameters) == dict):
        # ParsArray =[]
        # for i in range(len(Parameters)):
        #     ParsArray.append(Parameters[ParsMeta[i]]) 
        # ParsArray = DictToArray(Parameters,ParsMeta)
        
        # If any values are missing, they will be set to 1
        ParsArray = DictToArray(Parameters,ParsMeta,np.ones(len(ParsMeta)))
    else: 
        ParsArray = Parameters
        
    # InitArray = list(InitialConditions.values())
    # ParsArray = list(Parameters.values())
    # InitArray =InitialConditions
    # ParsArray =Parameters
    
    
    sol = solve_ivp(ModelFunction,[t0,tEnd],InitArray,t_eval=TimeRange,args=ParsArray,rtol=1e-10,atol=1e-10)
    # sol = solve_ivp(ModelFunction,[t0,tEnd],InitArray,t_eval=TimeRange,args=ParsArray)
    
    return sol.y
    
def prettyDict(curDict):
    toReturn = ''
    for key in curDict:
        toReturn += f'{key}: {curDict[key]}, '
    return toReturn[:-2]
    
## Simulation schemes
class Change:
    def __init__(self,tChange,AddVariables = [],MultiplyVariables = [],SetVariables = [],AddParameters = [],MultiplyParameters = [],SetParameters = []):
        self.t = tChange 
        self.AddVariables = AddVariables
        self.MultiplyVariables = MultiplyVariables
        self.SetVariables = SetVariables
        self.AddParameters = AddParameters
        self.MultiplyParameters = MultiplyParameters
        self.SetParameters = SetParameters
        
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
                
        if (len(self.SetVariables) > 0):
            if type(self.SetVariables) == dict:
                curStr += f'set variables {prettyDict(self.SetVariables)}, '
            else:
                curStr += f'set variables {self.SetVariables}, '

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
                
        if (len(self.SetParameters) > 0):
            if type(self.SetParameters) == dict:
                curStr += f'set parameters {prettyDict(self.SetParameters)}, '
            else:
                curStr += f'set parameters {self.SetParameters}, '
        # Remove excess comma and space
        curStr = curStr[:-2]
        return curStr 
    
    def __copy__(self):
        return type(self)(self.t,self.AddVariables,self.MultiplyVariables,self.SetVariables,self.AddParameters,self.MultiplyParameters,self.SetParameters)
        
    def copy(self):
        return copy(self)
        
    
class Scheme:
    def __init__(self,ModelName,InitialConditions,Parameters,tStart,tEnd,Changes=None):
        self.ModelName = ModelName
        self.InitialConditions = InitialConditions
        self.Parameters = Parameters
        self.tStart = tStart
        self.tEnd = tEnd
        
        # # Get reference to model
        # self.Model,self.ModelMeta = getModel(ModelName)
        
        # Initialize changes
        if (Changes == None):
            self.Changes = []
        else: 
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
        
        _,ModelMeta = getModel(self.ModelName)
        VarsMeta,ParsMeta = ModelMeta
        # If there are no changes, simply simulate
        if (len(self.Changes) == 0):
            totT = np.linspace(self.tStart,self.tEnd,tRes)
            totResult = simulateModel(self.ModelName,TimeRange=totT,InitialConditions=self.InitialConditions,Parameters=self.Parameters)
            
            # # Add simulation to object
            # self.result = type('SimulationResult', (), {})()
            # self.result.t = tRange
            # self.result.y = simulationOutput
        # If there are changes, simply go through them
        # (Assuming they are in correct order. Changes should be ordered using sortChanges())
        else:
            tInit = self.tStart
            curInit = self.InitialConditions.copy()
            curPars = self.Parameters.copy()
            
            # For each change, run until the change time
            for i in range(len(self.Changes)):
                curChange = self.Changes[i]
                
                curT = np.linspace(tInit,curChange.t,tRes) 
                
                # Run period
                simulationOutput = simulateModel(self.ModelName,TimeRange=curT,InitialConditions=curInit,Parameters=curPars)
                
                # Get final values
                finalState = simulationOutput[:,-1]
                # Add variables
                # if (type(curChange.AddVariables) == dict):
                #     finalState = finalState + DictToArray(curChange.AddVariables,VarsMeta)
                if (len(curChange.AddVariables) > 0):
                    if (type(curChange.AddVariables) == dict):
                        # finalState = finalState + DictToArray(curChange.AddVariables,VarsMeta) 
                        
                        # If any values are missing, they will be set to zero
                        VarsArray = DictToArray(curChange.AddVariables,VarsMeta,np.zeros(len(VarsMeta)))
                        
                        finalState = finalState + VarsArray
                        
                    else:
                        finalState = finalState + curChange.AddVariables
                # Multiply variables
                if (len(curChange.MultiplyVariables) > 0):
                    if (type(curChange.MultiplyVariables) == dict):
                        # toMult = DictToArray(curChange.MultiplyVariables,VarsMeta) 
                        
                        # If any values are missing, they will be set to one
                        toMult = DictToArray(curChange.MultiplyVariables,VarsMeta,np.ones(len(VarsMeta)))
                    else:
                        toMult = curChange.MultiplyVariables
                    for j in range(len(finalState)):
                        finalState[j] = finalState[j] * toMult[j]
                        
                # Set variables
                if (len(curChange.SetVariables) > 0):
                    if (type(curChange.SetVariables) == dict):                        
                        # If any values are missing, they will be set to current values
                        toSet = DictToArray(curChange.SetVariables,VarsMeta,curInit)
                    else:
                        toSet = curChange.SetVariables
                    for j in range(len(finalState)):
                        finalState[j] =  toSet[j]
                        
                # Add to parameters
                if (len(curChange.AddParameters) > 0):
                    if (type(curPars) == dict):
                        # for key in curPars:
                        for key in curChange.AddParameters:
                            curPars[key] = curPars[key] + curChange.AddParameters[key]
                    else:
                        
                        if (type(curChange.AddParameters) == dict):
                            # curPars = curPars + DictToArray(curChange.AddParameters,ParsMeta) 
                            # If any values are missing, they will be set to zero
                            toAdd = DictToArray(curChange.AddParameters,ParsMeta,np.zeros(len(ParsMeta)))
                            # curPars = curPars + curArray
                        else: # If a list of values
                            toAdd = curChange.AddParameters
                            # curPars = curPars + curChange.AddParameters
                        for j in range(len(curPars)):
                            curPars[j] = curPars[j] + toAdd[j]
                # Multiply Parameters
                if (len(curChange.MultiplyParameters) > 0):
                    if (type(curChange.MultiplyParameters) == dict):
                        # toMult = DictToArray(curChange.MultiplyParameters,ParsMeta) 
                        
                        # If any values are missing, they will be set to one
                        toMult = DictToArray(curChange.MultiplyParameters,ParsMeta,np.ones(len(ParsMeta)))
                    else:
                        toMult = curChange.MultiplyParameters
                    
                    if (type(curPars) == dict):
                        # for key in curPars:
                        for key in curChange.MultiplyParameters:
                            curPars[key] = curPars[key] * curChange.MultiplyParameters[key]
                    else:
                        for j in range(len(curPars)):
                            curPars[j] = curPars[j] * toMult[j]
                            
                # Set Parameters
                if (len(curChange.SetParameters) > 0):
                    if (type(curChange.SetParameters) == dict):                        
                        # If any values are missing, they will be set to current values
                        toSet = DictToArray(curChange.SetParameters,ParsMeta,curPars)
                    else:
                        toSet = curChange.SetParameters
                    
                    if (type(curPars) == dict):
                        # for key in curPars:
                        for key in curChange.SetParameters:
                            curPars[key] = curChange.SetParameters[key]
                    else:
                        for j in range(len(curPars)):
                            curPars[j] = toSet[j]
                
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
        # Add results as individual attributes
        for i in range(len(VarsMeta)):
            curName = VarsMeta[i] 
            setattr(self.result,curName,totResult[i,:])
            
    def plot(self,fig=[],linestyle='-',color='k',showChanges=True,describeChanges=True):
        
        # If simulation has not been run, run it now
        # if (~hasattr(self,'result')):
        if (hasattr(self,'result') == False):
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
        
    def simulateAndReturnResults(self):
        self.simulate()
        return self.result
        
    # Helper functions for returning callable function
    def setDictAndSimulate(self,dictToChange,valToChange,value):
        dictToChange[valToChange] = value 
        return self.simulateAndReturnResults()
    def setInitAndSimulate(self,strToChange,value):
        # varsMeta,parsMeta = getModel(self.ModelName)[1]
        if strToChange in self.InitialConditions:
            self.InitialConditions[strToChange] = value 
        if strToChange in self.Parameters:
            self.Parameters[strToChange] = value 
        return self.simulateAndReturnResults()
        
        
    
    def getCallableFunction(self,toOptimize):
        # So far, only one parameter can be changed at a time, and full result-object is returned
        
        # If toOptimize is a string, it should refer to a initial condition or a starting parameter
        if (type(toOptimize) == str):
            funcToReturn = lambda x: self.setInitAndSimulate(toOptimize,x)
            return funcToReturn
        else:
            # If toOptimize is not a string, it assumed to be a list, ordered like so: 
            # 0: Which "Change" number to change something in. 
            # 1: Name of the Change-Dict 
            # 2: Name of the variable or parameter to change
            changeToOptimize = getattr(self,'Changes')[toOptimize[0]]
            dictToOptimize = getattr(changeToOptimize,toOptimize[1])
            
            funcToReturn = lambda x: self.setDictAndSimulate(dictToOptimize,toOptimize[2],x)
            return funcToReturn