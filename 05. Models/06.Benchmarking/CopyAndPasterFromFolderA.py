import os
import pandas as pd
import time

InitialTime = time.time()

DirName  = os.getcwd()

CaseName_Base = 'IEEE118_mod1'

Folder_A = 'A.The_full_year_MILP'
Folder_B = 'B.Operation_cost'
Folder_D = 'D.Representative_days_based_on_RES_and_Demand'
Folder_E = 'E.Representative_days_based_on_Line_Benefits_OptModel'
Folder_F = 'F.Representative_days_based_on_Line_Benefit_NN_OC_fy_1'
Folder_G = 'G.Representative_days_based_on_Line_Benefit_NN_OC_fy_2'
Folder_H = 'H.Representative_days_based_on_Line_Benefit_NN_OC_fy_3'
Folder_I = 'I.Representative_days_based_on_Line_Benefit_NN_OC_fy_4'
Folder_K = 'K.Investments_per_hour'
Folder_L = 'L.Cont_Investments_per_hour'
Folder_M = 'M.Random_forest'

FoldersToPaste = [Folder_B, Folder_D, Folder_E, Folder_K, Folder_L]
# FoldersToPaste = [Folder_M]

# Defining the case CaseName_ByStages plus the CaseName_ByStages_nc#
CasesToPaste = []
CasesToPaste.append(CaseName_Base + '_ByStages')

# RangeClusters = [i for i in range(10, 101, 10)] + [i for i in range(150, 401, 50)] + [1000, 2000, 4000]
RangeClusters = [110, 120, 130, 140]

for i in RangeClusters:
    CasesToPaste.append(CaseName_Base + '_ByStages_nc' + str(i))

CasesTime = time.time() - InitialTime
StartTime = time.time()
print('The time for defining the cases is ' + str(CasesTime) + ' seconds')


# Validating if the CasesToPaste exist in FoldersToPaste
for case in CasesToPaste:
    for folder in FoldersToPaste:
        output_directory_MAIN = DirName + '/' + folder + '/' + case + '/'
        output_directory_Set  = DirName + '/' + folder + '/' + case + '/1.Set/'
        output_directory_Par  = DirName + '/' + folder + '/' + case + '/2.Par/'
        output_directory_Out  = DirName + '/' + folder + '/' + case + '/3.Out/'
        if not os.path.exists(output_directory_MAIN):
            os.makedirs(output_directory_MAIN)
        if not os.path.exists(output_directory_Set):
            os.makedirs(output_directory_Set)
        if not os.path.exists(output_directory_Par):
            os.makedirs(output_directory_Par)
        if not os.path.exists(output_directory_Out):
            os.makedirs(output_directory_Out)

ExitingFileTime = time.time() - StartTime
StartTime = time.time()
print('The time for validating the cases is ' + str(ExitingFileTime) + ' seconds')

# reading all the csv files from Folder A, subfolder 1.Set and  2.Par and saving in a dictionary
dict_Set = {}
dict_Par = {}
for file in os.listdir(os.path.join(DirName, Folder_A, CaseName_Base, '1.Set')):
    if file.endswith(".csv"):
        dict_Set[file] = pd.read_csv(os.path.join(DirName, Folder_A, CaseName_Base, '1.Set', file), header=None)
        dict_Set[file].fillna("", inplace=True)
for file in os.listdir(os.path.join(DirName, Folder_A, CaseName_Base, '2.Par')):
    if file.endswith(".csv"):
        dict_Par[file] = pd.read_csv(os.path.join(DirName, Folder_A, CaseName_Base, '2.Par', file), header=None)
        dict_Par[file].fillna("", inplace=True)

ReadingFilesTime = time.time() - StartTime
StartTime = time.time()
print('The time for reading the files is ' + str(ReadingFilesTime) + ' seconds')

# saving the dataframes in the new folders with a new name
for folder in FoldersToPaste:
    for case in CasesToPaste:
        for file in dict_Set:
            print('Copying and pasting the file ' + file + ' from ' + CaseName_Base + ' to ' + case + ' in ' + folder)
            dict_Set[file].fillna("", inplace=True)
            dict_Set[file].to_csv(os.path.join(DirName, folder, case, '1.Set', file.split('_', 3)[0]+'_'+file.split('_', 3)[1]+'_'+file.split('_', 3)[2]+'_'+case+'.csv'), index=False, header=False)
        for file in dict_Par:
            print('Copying and pasting the file ' + file + ' from ' + CaseName_Base + ' to ' + case + ' in ' + folder)
            dict_Par[file].fillna("", inplace=True)
            dict_Par[file].to_csv(os.path.join(DirName, folder, case, '2.Par', file.split('_', 3)[0]+'_'+file.split('_', 3)[1]+'_'+file.split('_', 3)[2]+'_'+case+'.csv'), index=False, header=False)
        path_to_ComputationTime_file = os.path.join(DirName, folder, case, '3.Out', 'ComputationTime.txt')
        with open(path_to_ComputationTime_file, 'w') as f:
            f.write(str(0.0))

SavingFilesTime = time.time() - StartTime
StartTime = time.time()
print('The time for saving the files is ' + str(SavingFilesTime) + ' seconds')

elapsed_time = round(time.time() - InitialTime)
print('Elapsed time: {} seconds'.format(elapsed_time))
