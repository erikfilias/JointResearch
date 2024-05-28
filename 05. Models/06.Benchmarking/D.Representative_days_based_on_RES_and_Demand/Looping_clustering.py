import os
import pandas as pd
import time
import InputBasedClustering_Modules as ibc

InitialTime = time.time()

DirName  = os.getcwd()

CaseName_Base = '9n_mod1'

# Defining the case CaseName_ByStages plus the CaseName_ByStages_nc#
CasesToPaste = []
# CasesToPaste.append(CaseName_Base + '_ByStages')

RangeClusters = [i for i in range(10, 101, 10)] + [i for i in range(150, 401, 50)]
# RangeClusters = [1000]

for i in RangeClusters:
    CasesToPaste.append(CaseName_Base + '_ByStages_nc' + str(i))

CasesTime = time.time() - InitialTime
StartTime = time.time()
print('The time for defining the cases is ' + str(CasesTime) + ' seconds')

for i in RangeClusters:
    print('Clustering of case ' + CaseName_Base + '_ByStages_nc' + str(i) + ' is starting')
    ibc.main(0, DirName, i, CaseName_Base)
    ClusteringTime = time.time() - StartTime
    StartTime = time.time()
    print('The time for clustering the case ' + CaseName_Base + '_ByStages_nc' + str(i) + ' is ' + str(ClusteringTime) + ' seconds')

elapsed_time = round(time.time() - InitialTime)
print('Elapsed time: {} seconds'.format(elapsed_time))
