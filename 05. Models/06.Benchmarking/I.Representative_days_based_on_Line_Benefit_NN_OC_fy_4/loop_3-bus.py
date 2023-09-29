import os

CASE   = '3-bus_ByStages_nc20'

path_to_scan = os.path.dirname(__file__)
filter = '3-bus_ByStages_nc'

directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if f.is_dir() and filter in os.path.basename(f.path)]
print(directories)

for case in directories[]:
    command = f"python oSN_Main_v2.py --case {CASE}"
    print(command)
    #os.system(command)