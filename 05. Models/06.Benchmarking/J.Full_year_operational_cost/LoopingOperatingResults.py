import os
import pandas as pd
import time

InitialTime = time.time()

DirName  = os.getcwd()

CaseName_Base = '9n'

Cases = ['9n', '9n_mod1', 'RTS24', 'RTS24_mod1', 'RTS24_mod2', 'IEEE118', 'IEEE118_mod1']

min_nb = 10
max_nb = 400

origin_folders=["B", "D", "E", "K", "L"]
# origin_folders=["L"]

# Loop over each origin folder
for case in Cases:
    print(f'Running for case:  {case}')
    for origin_folder in origin_folders:
        print(f'Running for origin_folder:  {origin_folder}')
        command = f"python loop_operational_in_mem.py --case {case} --origin_folder {origin_folder} --min_nb {min_nb} --max_nb {max_nb}"
        print(command)
        os.system(command)
# for origin_folder in origin_folders:
#     print(f'Running for origin_folder:  {origin_folder}')
#     command = f"python loop_operational_in_mem.py --case {CaseName_Base} --origin_folder {origin_folder} --min_nb {min_nb} --max_nb {max_nb}"
#     print(command)
#     os.system(command)
