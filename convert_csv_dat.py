
# coding: utf-8

# In[ ]:


import os
import csv

conversion_csv = 'Alternative_scenario_input_operation.csv'
conversion_dat = 'Alternative_scenario_input_operation.dat'

input_add = os.path.join(os.getcwd(), conversion_csv)
output_add = os.path.join(os.getcwd(), conversion_dat)

with open(input_add , 'r') as fin,      open(output_add, 'w') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout,fieldnames=["param:","tech:", "inv", "var", "fix",  "f", "capa", "eff", "em", "yr",  ":="], delimiter='\t')
        writer.writeheader()
        writer.writerows(reader)

