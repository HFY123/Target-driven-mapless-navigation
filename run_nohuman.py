import os
import subprocess
for j in range(1,3):
    file_name = 'oneturn1_nohuman' + str(j)
    path = r"train.py"
    p = subprocess.Popen(f"python {path} --model_name {file_name}", shell=True)
    p.wait()    

