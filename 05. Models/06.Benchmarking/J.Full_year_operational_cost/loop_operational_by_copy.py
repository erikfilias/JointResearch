import os
import sys
import argparse
import re

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case', type=str, default="3-bus")
parser.add_argument('--min_nb', type=int, default=None, help='Minimum value for the integer following "nc"')
parser.add_argument('--max_nb', type=int, default=None, help='Maximum value for the integer following "nc"')

args = parser.parse_args()

case = args.case
min_nb = args.min_nb
max_nb = args.max_nb

# path_to_scan = os.path.dirname("../D.Representative_days_based_on_RES_and_Demand/")
path_to_scan = os.path.dirname("../L.Cont_Investments_per_hour/")

filter = f'{case}_ByStages_nc'

directories = []

for f in os.scandir(path_to_scan):
    if f.is_dir():
        basename = os.path.basename(f.path)
        match = re.search(f'{filter}(\d+)', basename)
        if match:
            nb = int(match.group(1))
            if (min_nb is None or nb >= min_nb) and (max_nb is None or nb <= max_nb):
                directories.append(basename)

directories = sorted(directories, key=lambda x: int(re.search(r'nc(\d+)', x).group(1)))

print(directories)

# folder = "D.Representative_days_based_on_RES_and_Demand"
folder = "L.Cont_Investments_per_hour"

for subfolder in directories:
    print(subfolder)
    try:
        command_fix_investments= f"python CopyInvestmentDecisions.py --folder {folder} --subfolder {subfolder} "
        print(command_fix_investments)
        os.system(command_fix_investments)
    except Exception as e:
        # Handle the exception
        print(f"An error occurred: {e}")

        # Terminate the program
        sys.exit()


    command_operational_run = f"python oSN_Main_operational.py --case {case}"
    print(command_operational_run)
    os.system(command_operational_run)
