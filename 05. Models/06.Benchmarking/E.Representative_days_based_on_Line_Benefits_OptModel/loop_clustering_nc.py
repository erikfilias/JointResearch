import os

OrgCase = '9n'

path_to_scan = os.path.dirname(__file__)
filter = f'{OrgCase}_ByStages'

directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if f.is_dir() and filter in os.path.basename(f.path)]

print(directories)

for case in directories:
    if case == OrgCase+'_ByStages':
        # command = f"python OutputBasedClustering.py --cn {OrgCase} --case {case} --inc {'Yes'} --nc {20}"
        # print(command)
        # os.system(command)
        print(case)
    else:
        ncluster = int(case[len(filter) + 3:len(case)])
        command = f"python OutputBasedClustering.py --cn {OrgCase} --case {case} --inc {'No'} --nc {ncluster}"
        print(command)
        os.system(command)
