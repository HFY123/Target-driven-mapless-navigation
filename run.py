import os
import subprocess
dic = {0:'0.05human_5S_', 1:'0.10human_oneturn1_', 2:'0.20human_oneturn1_', 3:'0.30human_oneturn1_', 4:'0.50human_oneturn1_', 5:'0.80human_oneturn1_', 6:'1.00human_oneturn1_'}
# percentage = {0:'0.05', 1:'0.10', 2:'0.20', 3:'0.30', 4:'0.50', 5:'0.80'}
# for i in range(len(dic)):
for i in range(1):
    name = dic[i]
    per = dic[i][:4]
    for j in range(1,3):
        file_name = name + str(j)
        path = r"train2.py"
        p = subprocess.Popen(f"python {path} --model_name {file_name} --percentage {per}", shell=True)
        p.wait()    

