
# coding: utf-8

# In[ ]:


from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import Var
opt=SolverFactory('glpk')
model = AbstractModel()


model.T = Set()

model.tech = Set()

model.M = Set()


def Day_init(model):
    return range(1,int(len(model.T)/24))
model.Day = Set(initialize=Day_init)



# Parameter definition Demand, PV and wind hourly distribution

model.D = Param(model.T, within=NonNegativeReals)

model.PV = Param(model.T, within=NonNegativeReals)

model.wind = Param(model.T, within=NonNegativeReals)

model.nuclear = Param(model.T, within=NonNegativeReals)

model.average_wind_capacity = Param(within=NonNegativeReals)

model.average_PV_capacity = Param(within=NonNegativeReals)

model.inv = Param(model.tech, within=NonNegativeReals)

model.var = Param(model.tech, within = NonNegativeReals)

model.fix = Param(model.tech, within = NonNegativeReals)

model.f = Param(model.tech, within = NonNegativeReals)

model.capa = Param(model.tech, within = NonNegativeReals)

model.eff = Param(model.tech, within = NonNegativeReals)

model.em = Param(model.tech, within = NonNegativeReals)

model.yr = Param(model.tech, within = NonNegativeReals)

model.annualD = Param( within = NonNegativeReals)

model.waste_limit = Param(within = NonNegativeReals)

model.waste_min = Param(within = NonNegativeReals)

model.waste_max = Param(within = NonNegativeReals)

model.bio_limit = Param(within = NonNegativeReals)

model.bio_min= Param(within = NonNegativeReals)

model.bio_max = Param(within = NonNegativeReals)

model.fc_limit = Param(within = NonNegativeReals) 

model.fc_max = Param(within = NonNegativeReals) 

model.fc_min = Param(within = NonNegativeReals) 

model.pump_str = Param(within = NonNegativeReals)  
    
model.discount=Param(within=NonNegativeReals)

model.coal_ratio=Param(within=NonNegativeReals)

model.pumped_ratio=Param(within=NonNegativeReals)

model.over =Param(within=NonNegativeReals)

model.em_limit=Param(within=NonNegativeReals)

model.charge= Var(model.T, within=NonNegativeReals)

model.discharge= Var(model.T, within=NonNegativeReals)

model.dispatch = Var(model.T, model.tech, within=NonNegativeReals)
model.soc = Var(model.T, within=NonNegativeReals)
model.fuel = Var(model.T, model.tech, within=NonNegativeReals)
model.emission = Var(within=NonNegativeReals)
model.overproduction = Var(model.T, within=NonNegativeReals)

model.res_amount = Var(within=NonNegativeReals)
model.res_share  = Var(within=NonNegativeReals)

model.solar_p = Var(model.T, within=NonNegativeReals)
model.wt_p = Var(model.T, within=NonNegativeReals)

model.invbytech = Var(model.tech, within=NonNegativeReals)
model.fuelbytech = Var(model.tech, within=NonNegativeReals)
model.var_bytech = Var(model.tech, within=NonNegativeReals)
model.fixbytech = Var(model.tech, within=NonNegativeReals)

model.PV_var = Var(within=NonNegativeReals)
model.wind_var = Var(within=NonNegativeReals)
model.NUKE_fuel = Var(within=NonNegativeReals)

model.total = Var(within=NonNegativeReals)

def capa_annuity (model,  k) :
    return model.discount*model.inv[k]/(1-(1+model.discount)**(-model.yr[k]))
model.annuity = Param(model.tech,  initialize = capa_annuity)


def demand_t(model) :
    return sum(model.D[i] for i in model.T)
model.D_sum=Param(initialize = demand_t)


def demand_e (model,i) :
    return model.D[i]/model.D_sum
model.Dprofile = Param(model.T, initialize = demand_e)

def demand_e1 (model,i) :
    return model.Dprofile[i]*model.annualD*1000000
model.eldemand = Param(model.T, initialize = demand_e1)

def PV_max(model) :
    return max(model.PV[i] for i in model.T)
model.maxPV = Param(initialize = PV_max)

def PV_e (model, i) :
    return model.PV[i]/model.maxPV
model.PVprofile = Param(model.T, initialize = PV_e)

def wind_max(model) :
    return max(model.wind[i] for i in model.T)
model.maxwind = Param(initialize = wind_max)

def wind_e (model, i) :
    return model.wind[i]/model.maxwind
model.windprofile = Param(model.T, initialize = wind_e)

def nuclear_max(model) :
    return max(model.nuclear[i] for i in model.T)
model.maxnuclear = Param(initialize = nuclear_max)

def nuclear_e (model, i) :
    return model.nuclear[i]/model.maxnuclear 
model.nuclearprofile = Param(model.T, initialize = nuclear_e)

def nuclear_p (model, i):
    return model.nuclearprofile[i]*model.capa['nuke']*1000
model.nuke_p = Param(model.T, initialize = nuclear_p)


# Constraints

# charge discharge rule -pumped hydro
def bcap_rule(model,i) :
    return model.charge[i] <= model.capa['pumped']*1000 
model.bcap_charge = Constraint(model.T,rule=bcap_rule)

def bcap_rule1(model,i) :
    return model.discharge[i] <=  model.capa['pumped']*1000 
model.bcap_charge1 = Constraint(model.T,rule=bcap_rule1)

def dcap_rule(model,k)  :
    return sum(model.discharge[i] for i in sequence((24*(k-1)+1), 24*k)) <= model.capa['pumped']*5000 
model.dcap_limit = Constraint(model.Day, rule=dcap_rule)  

def dcap_rule1(model,k)  :
    return sum(model.charge[i] for i in sequence((24*(k-1)+1), 24*k)) <= model.capa['pumped']*5000 
model.dcap_limit1 = Constraint(model.Day, rule=dcap_rule1)  

#SOC capacity rule
def soc_capa_rule(model, i) :
    return model.soc[i] <= model.pump_str
model.soccapabattery = Constraint(model.T, rule=soc_capa_rule)

def pv_production (model,i) :
    return model.solar_p[i] == model.PVprofile[i]*(model.capa['solar'])*1000
model.pv_p = Constraint(model.T, rule=pv_production)

def wt_production (model, i) :
    return model.wt_p[i] == model.windprofile[i]*(model.capa['WT'])*1000
model.wind_p = Constraint(model.T, rule=wt_production)

# balance rule
def balance_rule(model, i) :
    return model.eldemand[i]== sum(model.dispatch[i,k] for k in model.tech)                          +model.solar_p[i]                          +model.wt_p[i]                          +model.nuke_p[i]                          -model.charge[i]                          +model.discharge[i]                          -model.overproduction[i]
model.balance = Constraint(model.T, rule=balance_rule)            

##NG regulation for wind and pv
def NGregulation_init(model,i):
    return model.dispatch[i,'ng'] >=0.15*(model.wt_p[i]+model.solar_p[i])
model.regulation_ng = Constraint(model.T, rule=NGregulation_init)


## coal regulation
def coal_regulation(model, i) :
    if i==1 :
        return model.dispatch[i,'coal']>=1
    else :
        return model.dispatch[i,'coal'] - model.dispatch[i-1,'coal'] <= 2000
model.coal_diff = Constraint(model.T, rule = coal_regulation)

def coal_regulation1(model, i) :
    if i==1 :
        return model.dispatch[i,'coal']>=1
    else :
        return model.dispatch[i,'coal'] - model.dispatch[i-1,'coal'] >= -2000
model.coal_diff1 = Constraint(model.T, rule = coal_regulation1)

#conversion rule
def con_rule(model, i, k) :
    return model.dispatch[i,k]== model.fuel[i,k]*model.eff[k]/100
model.conversion = Constraint(model.T, model.tech, rule=con_rule)

#capacity cosntraint for dispatchable plants
def dis_capacity_rule(model, i, k) :
    return model.dispatch[i,k] <= (model.capa[k])*1000
model.capacity_constraint = Constraint(model.T, model.tech, rule=dis_capacity_rule)


#emission rule
def emission_rule(model) :
    return model.emission == sum(model.em[k]*sum(model.fuel[i,k] for i in model.T) for k in model.tech)
model.emission_sum = Constraint(rule=emission_rule)

def emission_rule1(model) :
    return model.emission <= model.em_limit
model.emission_sum_limit = Constraint(rule=emission_rule1)

# SOC rule
def soc_rule(model, i) :
    if i ==1 :
        return model.soc[i]== 0.5*model.pump_str
    if i>=2 and i<= (len(model.T)-1) :
        return model.soc[i] == model.soc[i-1]+model.charge[i-1] - model.discharge[i-1]
    else :
        return model.soc[len(model.T)]== model.soc[len(model.T)-1]+model.charge[i-1] - model.discharge[i-1]   
model.sc = Constraint(model.T, rule=soc_rule)



# waste limitation
def waste_rule(model):
    return sum(model.dispatch[i, 'waste'] for i in model.T) <= model.waste_limit
model.waste_sum= Constraint(rule=waste_rule)

def waste_minimum_rule(model,i):
    return model.dispatch[i, 'waste']  >= model.waste_min
model.waste_floor= Constraint(model.T, rule=waste_minimum_rule)

def waste_maximum_rule(model,i):
    return model.dispatch[i, 'waste']  <= model.waste_max
model.waste_ceiling= Constraint(model.T, rule=waste_maximum_rule)

# bio limitation
def bio_rule(model):
    return sum(model.dispatch[i, 'bio'] for i in model.T) <= model.bio_limit
model.bio_sum= Constraint(rule=bio_rule)

def bio_minimum_rule(model, i):
    return model.dispatch[i, 'bio']  >= model.bio_min
model.bio_floor= Constraint(model.T, rule=bio_minimum_rule)

def bio_maximum_rule(model, i):
    return model.dispatch[i, 'bio']  <= model.bio_max
model.bio_ceiling= Constraint(model.T, rule=bio_maximum_rule)

# FC limitation
def fc_rule(model):
    return sum(model.dispatch[i, 'FC'] for i in model.T) <= model.fc_limit
model.fc_sum= Constraint(rule=fc_rule)

def fc_minimum_rule(model, i):
    return model.dispatch[i, 'FC']  >= model.fc_min
model.fc_floor= Constraint(model.T, rule=fc_minimum_rule)

def fc_maximum_rule(model, i):
    return model.dispatch[i, 'FC']  <= model.fc_max
model.fc_ceiling= Constraint(model.T, rule=fc_maximum_rule)

## rule for res share
def res_amount_rule(model):
    return model.res_amount == sum(model.PVprofile[i]*model.capa['solar']*1000                                   +model.windprofile[i]*(model.capa['WT'])*1000 for i in model.T) 
model.resamount = Constraint(rule=res_amount_rule)

def res_share_rule(model):
    return model.res_share == model.res_amount/(model.annualD*1000000)
model.resshare = Constraint(rule=res_share_rule)



#def res_rule(model):
#    return sum(model.PVprofile[i]*(model.capa['solar']+model.newcapa['solar'])*1000\
#               +model.windprofile[i]*(model.capa['WT']+model.newcapa['WT'])*1000 for i in model.T) == 0.19*model.annualD*1000000
#model.res_regulation = Constraint(rule=res_rule)



def coal_rule(model):
    return sum(model.dispatch[i,'coal'] for i in model.T) <=  model.coal_ratio*model.annualD*1000000
model.coal_regulation = Constraint(rule=coal_rule)

##overproduction rule
def over_rule(model):
    return sum(model.overproduction[i] for i in model.T) <= model.over*model.annualD*1000000
model.lose_regulation = Constraint(rule=over_rule)



# cost object function
def fixed_OM_rule(model,k):
    return model.fixbytech[k] == model.fix[k]*model.capa[k]*1000
model.exist_f= Constraint(model.tech, rule = fixed_OM_rule)

def invest_rule(model,k):
    return model.invbytech[k]==model.annuity[k]*model.capa[k]*1000
model.exist_inv= Constraint(model.tech, rule = invest_rule)

def fuelcost_rule(model,k) :
    return model.fuelbytech[k] == model.f[k]*sum(model.fuel[i,k] for i in model.T)
model.fueltotal = Constraint(model.tech, rule = fuelcost_rule)

def varcost_rule(model, k):
    return model.var_bytech[k] == model.var[k]*sum(model.dispatch[i,k] for i in model.T)
model.vartotal = Constraint(model.tech, rule = varcost_rule)

def varcostPV_rule (model) :
    return model.PV_var == sum(model.PVprofile[i]*(model.capa['solar'])*1000*model.var['solar'] for i in model.T)
model.PVvar = Constraint( rule = varcostPV_rule)

def varcostWT_rule (model) :
    return model.wind_var == sum(model.windprofile[i]*(model.capa['WT'])*1000*model.var['WT'] for i in model.T)
model.WTvar = Constraint( rule = varcostWT_rule)

def Nukefuel_rule (model) :
    return model.NUKE_fuel == sum(model.nuclearprofile[i]*model.capa['nuke']*1000 for i in model.T)*model.f['nuke']
model.NUKE_cost = Constraint(rule = Nukefuel_rule)



def totalcost_rule (model) :
    return model.total == sum(model.f[k]*sum(model.fuel[i,k] for i in model.T) for k in model.tech)+sum(model.fix[k]*model.capa[k]*1000 for k in model.tech)+sum(model.annuity[k]*model.capa[k]*1000 for k in model.tech)+sum(model.var[k]*sum(model.dispatch[i,k] for i in model.T) for k in model.tech)+sum(model.PVprofile[i]*(model.capa['solar'])*1000*model.var['solar']     +model.windprofile[i]*(model.capa['WT'])*1000*model.var['WT'] for i in model.T)+sum(model.nuclearprofile[i]*model.capa['nuke']*1000 for i in model.T)*model.f['nuke']
model.total_cost = Constraint( rule=totalcost_rule)
    

def cost_rule(model):
    return sum(model.f[k]*sum(model.fuel[i,k] for i in model.T) for k in model.tech)+sum(model.fix[k]*model.capa[k]*1000 for k in model.tech)+sum(model.annuity[k]*model.capa[k]*1000 for k in model.tech)+sum(model.var[k]*sum(model.dispatch[i,k] for i in model.T) for k in model.tech)+sum(model.PVprofile[i]*(model.capa['solar'])*1000*model.var['solar']     +model.windprofile[i]*(model.capa['WT'])*1000*model.var['WT'] for i in model.T)+sum(model.overproduction[i]*model.f['ng']*10 for i in model.T)+sum(model.nuclearprofile[i]*model.capa['nuke']*1000 for i in model.T)*model.f['nuke']
model.cost = Objective(rule = cost_rule)

instance = model.create_instance('Alternative_scenario_input_operation.dat', report_timing=True)

results = opt.solve(instance, tee=True)
results.write()
instance.solutions.load_from(results)

show0=[];
show1=[];
show2=[];
show3=[];

for v in instance.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(instance, str(v))
    show0.append(str(v))
    show1.append(str(v))
    show2.append(str(v))
    for i in varobject:
        show1.append(i)
        show2.append(varobject[i].value)



        
from pyomo.core import Param        
showPara=[]; 
for p in instance.component_objects(Param, active=True):
    parmobject = getattr(instance,str(p))
    showPara.append(str(p))
    for i in parmobject:
        showPara.append(parmobject[i])


# coding: utf-8

# In[ ]:


k=[]
dispatch=[]


for i in range(len(show1)) : 
    if type(show1[i])== tuple :
        k.append(i)
        dispatch.append(show1[i]) 
        

time1=[]
tech1=[]
for i in range(len(dispatch)) :
    time, tech = dispatch[i]
    time1.append(time)
    tech1.append(tech)      

import math
## in order remove NAN "math.isnan" function is used
matrix = []
fueltech = []
dispatchtech = []
othertech = []
annu = []
cost = []
k=0
m=0

for i in range(len(show1)):
    if show1[i] in show0 :
        f=show1[i]
        matrix.append([show1[i], f, None, show2[i]])
        othertech.append([show1[i], f, None, show2[i]])
    elif type(show1[i]) == tuple :
        matrix.append([time1[k], f, tech1[k],  show2[i]])
        if f == 'dispatch':
            dispatchtech.append([time1[k], tech1[k], show2[i]])
        elif f == 'fuel':
            fueltech.append([time1[k], tech1[k], show2[i]])
        k+=1
    else  : 
        matrix.append([show1[i], f, None, show2[i]])
        if type(show1[i]) == int  and  math.isnan(show1[i])==False :
            annu.append([show1[i], f,  show2[i]])            
        else:
            cost.append([show1[i], f, show2[i]])    

import numpy as np
import csv
import pandas as pd

df_show1=pd.DataFrame(show1)
df_show2=pd.DataFrame(show2)

df_showpara= pd.DataFrame(showPara)
df_matrix = pd.DataFrame(matrix)
df_fueltech = pd.DataFrame(fueltech)
df_dispatchtech = pd.DataFrame(dispatchtech)
df_othertech = pd.DataFrame(othertech)
df_annu = pd.DataFrame(annu)
df_cost = pd.DataFrame(cost)
paramatrix=[]
for i in range(len(showPara)) :
    if type(showPara[i])==str:
        f=showPara[i]
        k=1
    else :
        paramatrix.append([f,k,showPara[i]])
        k+=1

df_paramatrix=pd.DataFrame(paramatrix)
#df_paramatrix.to_csv('paramatrix.csv', index=False, Header=False)


df_paramatrix.rename(
    columns={
        0 : 'Parameter',
        1 : 'hour',
        2 : 'value'
        }, inplace=True )
df_paramatrix

df_paramatrix_pivot = df_paramatrix.pivot(index = 'hour', columns = 'Parameter', values='value' )



df_matrix.rename(
    columns={
        0 : 'hour',
        1 : 'tech1',
        2 : 'tech2',
        3 : 'value'
        }, inplace=True )

df_annu.rename(
    columns={
        0 : 'hour',
        1 : 'tech1',
        2 : 'value'
        }, inplace=True )

df_fueltech.rename(
    columns={
        0 : 'hour',
        1 : 'tech2',
        2 : 'value'
        }, inplace=True )

df_dispatchtech.rename(
    columns={
        0 : 'hour',
        1 : 'tech2',
        2 : 'value'
        }, inplace=True )

df_cost.rename(
    columns={
        0 : 'tech',
        1 : 'item',
        2 : 'value'
        }, inplace=True )
df_cost = df_cost[['item', 'tech','value']]

df_cost_pivot = df_cost.pivot(index = 'tech', columns = 'item', values='value' )

df_cost_pivot



df_annupivot=df_annu.pivot(index = 'hour', columns = 'tech1', values='value' )



df_dispatchtechpivot=df_dispatchtech.pivot(index = 'hour', columns = 'tech2', values='value' )


df_dispatchtechpivot.coal
df_dispatchtechpivot.ng
df_dispatchtechpivot.bio
df_dispatchtechpivot.waste

result = pd.concat([df_paramatrix_pivot.eldemand, df_paramatrix_pivot.nuke_p, df_annupivot.solar_p, df_annupivot.wt_p,                   df_dispatchtechpivot.coal, df_dispatchtechpivot.ng, df_dispatchtechpivot.bio, df_dispatchtechpivot.waste,                   df_dispatchtechpivot.FC,df_annupivot.charge, df_annupivot.discharge, df_annupivot.overproduction ],                           axis=1)

df_cost_pivot = df_cost.pivot(index = 'tech', columns = 'item', values='value' )
df_cost_pivot.fuelbytech.nuke=df_cost_pivot.NUKE_fuel[0]
df_cost_pivot.var_bytech.WT=df_cost_pivot.wind_var[0]
df_cost_pivot.var_bytech.solar=df_cost_pivot.PV_var[0]
cost_by_tech=pd.concat([df_cost_pivot.fuelbytech, df_cost_pivot.var_bytech,                       df_cost_pivot.fixbytech, df_cost_pivot.invbytech,                       df_cost_pivot.total, df_cost_pivot.res_amount,                       df_cost_pivot.res_share,df_cost_pivot.emission], axis=1)

cost_by_tech.to_csv('cost_RESshare_emission.csv', index = True, header = True)
result.to_csv('result_operation.csv', index=False, header=True)





