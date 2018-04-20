
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

hourlyvalue=pd.read_csv('result_operation.csv')
X=np.arange(len(hourlyvalue.eldemand))
D_el = hourlyvalue.eldemand.astype(np.float64)
Nuke_el = hourlyvalue.nuke_p.astype(np.float64)
PV_el = hourlyvalue.solar_p.astype(np.float64)
Wind_el = hourlyvalue.wt_p.astype(np.float64)
Coal_el = hourlyvalue.coal.astype(np.float64)
NG_el = hourlyvalue.ng.astype(np.float64)
Other_el = hourlyvalue.bio.astype(np.float64) + hourlyvalue.waste.astype(np.float64) + hourlyvalue.FC.astype(np.float64)
Ch_el = hourlyvalue.charge.astype(np.float64)
Dis_el = hourlyvalue.discharge.astype(np.float64)
Over_el = hourlyvalue.overproduction.astype(np.float64)
Net_el = Dis_el-Ch_el
positive_el = []
negative_el = []
for i in range(0, len(Net_el)):
    if Net_el[i]>0:
        positive_el.append(Net_el[i])
        negative_el.append(0)
    elif Net_el[i]==0:
        positive_el.append(0)
        negative_el.append(0)
    else :
        positive_el.append(0)
        negative_el.append(Net_el[i])






fig, ax=plt.subplots()
plt.plot([],[], color = 'm', label = 'Nuclear', linewidth=5)
plt.plot([],[], color = 'y', label = 'Coal', linewidth=5)
plt.plot([],[], color = 'c', label = 'NG', linewidth=5)
plt.plot([],[], color = 'r', label = 'Other', linewidth=5)
plt.plot([],[], color = 'g', label = 'PV', linewidth=5)
plt.plot([],[], color = 'b', label = 'Wind', linewidth=5)
plt.plot([],[], color = 'k', label = 'Discharge', linewidth=5)
plt.plot(X,-Over_el, label = 'Surplus', linewidth=1)
plt.plot(X,negative_el, label = 'charge', linewidth=1)

ax.stackplot(X, Nuke_el, Coal_el ,  NG_el , Other_el , PV_el, Wind_el, positive_el, colors=['m','y','c','r', 'g', 'b','k'])
ax.plot(X, D_el, 'r--', linewidth=3)
ax.stackplot(X, negative_el,-Over_el, colors=['y','b'])


begin=3864
end=4032
spacing = 24
minorLocator = MultipleLocator(spacing)
ax.set_xlim([begin,end])
ax.set_xlabel('hour')
ax.set_ylabel('MWh/h')
ax.xaxis.set_minor_locator(minorLocator)
plt.xticks(np.arange(begin, end, 24))
plt.grid()
plt.title('Electricity Balance')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('hourlybalance.png',bbox_inches='tight')
plt.show()
