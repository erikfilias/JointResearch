import os
import argparse
import re

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case', type=str, default=None)
parser.add_argument('--min_nb', type=int, default=None, help='Minimum value for the integer following "nc"')
parser.add_argument('--max_nb', type=int, default=None, help='Maximum value for the integer following "nc"')

args = parser.parse_args()

case = args.case
min_nb = args.min_nb
max_nb = args.max_nb

path_to_scan = os.path.dirname(__file__)
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



for case in directories:
    command = f"python oSN_Main_DCOPF_WoPCA.py --case {case}"
    print(command)
    os.system(command)
