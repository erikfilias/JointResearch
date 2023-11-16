import os
import pandas as pd
import time
import InputBasedClustering_Modules as ibc

InitialTime = time.time()

DirName  = os.getcwd()

CaseName_Base = '3-bus'

Folder_A = 'A.The_full_year_MILP'
Folder_D = 'D.Representative_days_based_on_RES_and_Demand'
Folder_E = 'E.Representative_days_based_on_Line_Benefits_OptModel'
Folder_F = 'F.Representative_days_based_on_Line_Benefit_NN_OC_fy_1'
Folder_G = 'G.Representative_days_based_on_Line_Benefit_NN_OC_fy_2'
Folder_H = 'H.Representative_days_based_on_Line_Benefit_NN_OC_fy_3'
Folder_I = 'I.Representative_days_based_on_Line_Benefit_NN_OC_fy_4'

FoldersToPaste = [Folder_D, Folder_E, Folder_F, Folder_G, Folder_H, Folder_I]

# Defining the case CaseName_ByStages plus the CaseName_ByStages_nc#
CasesToPaste = []
CasesToPaste.append(CaseName_Base + '_ByStages')

RangeClusters = [i for i in range(10, 101, 10)] + [1000, 2000, 4000]

for i in RangeClusters:
    CasesToPaste.append(CaseName_Base + '_ByStages_nc' + str(i))

CasesTime = time.time() - InitialTime
StartTime = time.time()
print('The time for defining the cases is ' + str(CasesTime) + ' seconds')

for i in RangeClusters:
    ibc.main(0, DirName, i, CaseName_Base)


SavingFilesTime = time.time() - StartTime
StartTime = time.time()
print('The time for saving the files is ' + str(SavingFilesTime) + ' seconds')

elapsed_time = round(time.time() - InitialTime)
print('Elapsed time: {} seconds'.format(elapsed_time))
