import os
import argparse

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)
parser.add_argument('--highnb',   type=str, default=False)

args = parser.parse_args()

case = 'IEEE118'
highnb = args.highnb

path_to_scan = os.path.dirname(__file__)
filter = f'{case}_ByStages_nc'
filter_high_nb = f"000"

# Defining the case CaseName_ByStages plus the CaseName_ByStages_nc#
CasesToPaste = []

RangeClusters = [i for i in range(150, 301, 50)]

for i in RangeClusters:
    CasesToPaste.append(str(case) + '_ByStages_nc' + str(i))

print(CasesToPaste)
for case in CasesToPaste:
    command = f"python oSN_Main_DCOPF.py --case {case}"
    print(command)
    os.system(command)
