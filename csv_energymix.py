
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


hourlyvalue=pd.read_csv('result_operation.csv')
Nuke_el = hourlyvalue.nuke_p.astype(np.float64)
PV_el = hourlyvalue.solar_p.astype(np.float64)
Wind_el = hourlyvalue.wt_p.astype(np.float64)
Coal_el = hourlyvalue.coal.astype(np.float64)
NG_el = hourlyvalue.ng.astype(np.float64)
Other_el = hourlyvalue.bio.astype(np.float64) + hourlyvalue.waste.astype(np.float64) + hourlyvalue.FC.astype(np.float64)
Ch_el = hourlyvalue.charge.astype(np.float64)
Dis_el = hourlyvalue.discharge.astype(np.float64)
#df_paramatrix_pivot.eldemand
Waste_dispat=hourlyvalue.bio.sum()        +hourlyvalue.waste.sum()
labels = 'nuclear', 'PV', 'Wind', 'coal', 'Natural gas', 'other'
sizes = [hourlyvalue.nuke_p.sum(), hourlyvalue.solar_p.sum(), hourlyvalue.wt_p.sum(),        Coal_el.sum(), NG_el.sum(),Other_el.sum()]
colors = ['lightcoral', 'green', 'skyblue', 'brown', 'yellowgreen', 'purple']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.title("Energy mix 2030")
plt.savefig('Energymix2030.png')
plt.show()

