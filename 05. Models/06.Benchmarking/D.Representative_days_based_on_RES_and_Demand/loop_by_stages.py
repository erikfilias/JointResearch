import os
import argparse

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)
parser.add_argument('--highnb',   type=str, default=False)

args = parser.parse_args()

case = args.case
highnb = args.highnb

path_to_scan = os.path.dirname(__file__)
filter = f'{case}_ByStages'
filter_high_nb = f"000"

if highnb:
    directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if
                   f.is_dir() and filter in os.path.basename(f.path) and (
                               filter_high_nb in os.path.basename(f.path))]
else:
    directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if
                   f.is_dir() and filter in os.path.basename(f.path) and not (
                               filter_high_nb in os.path.basename(f.path))]

print(directories)

for case in directories:
    command = f"python oSN_Main_DCOPF.py --case {case}"
    print(command)
    os.system(command)
