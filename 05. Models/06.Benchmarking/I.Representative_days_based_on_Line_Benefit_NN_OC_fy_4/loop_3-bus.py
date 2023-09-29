import os


path_to_scan = os.path.dirname(__file__)
filter = '3-bus_ByStages_nc'

directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if f.is_dir() and filter in os.path.basename(f.path)]
print(directories[2:4])

for case in directories[2:4]:
    command = f"python oSN_Main_v2.py --case {case}"
    print(command)
    os.system(command)