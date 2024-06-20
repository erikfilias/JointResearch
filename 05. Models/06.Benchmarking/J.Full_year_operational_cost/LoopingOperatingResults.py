import os
import pandas as pd
import time

InitialTime = time.time()

DirName  = os.getcwd()

CaseName_Base = 'RTS24_mod2'

min_nb = 10
max_nb = 400

origin_folders=["B", "D", "E", "K", "L"]
# origin_folders=["L"]

# Loop over each origin folder
for origin_folder in origin_folders:
    print(f'Running for origin_folder:  {origin_folder}')
    command = f"python loop_operational_in_mem.py --case {CaseName_Base} --origin_folder {origin_folder} --min_nb {min_nb} --max_nb {max_nb}"
    print(command)
    os.system(command)
