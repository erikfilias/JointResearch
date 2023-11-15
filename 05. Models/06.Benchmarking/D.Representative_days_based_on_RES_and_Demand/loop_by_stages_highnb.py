import os
import argparse

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)

args = parser.parse_args()

case = args.case

path_to_scan = os.path.dirname(__file__)
filter = f'{case}_ByStages'
filter_high_nb = f"000"

directories = [os.path.basename(f.path) for f in os.scandir(path_to_scan) if f.is_dir() and filter in os.path.basename(f.path) and filter_high_nb in os.path.basename(f.path)]


for case in directories:
    command = f"python oSN_Main_DCOPF.py --case {case}"
    print(command)
    # os.system(command)