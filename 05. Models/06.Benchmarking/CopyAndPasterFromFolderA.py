import os
import pandas as pd

DirName  = os.getcwd()

CaseName_Base = 'RTS24'

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

# reading all the csv files from Folder A, subfolder 1.Set and  2.Par and saving in a dictionary
dict_Set = {}
dict_Par = {}
for file in os.listdir(os.path.join(DirName, Folder_A, CaseName_Base, '1.Set')):
    if file.endswith(".csv"):
        dict_Set[file] = pd.read_csv(os.path.join(DirName, Folder_A, CaseName_Base, '1.Set', file))
for file in os.listdir(os.path.join(DirName, Folder_A, CaseName_Base, '2.Par')):
    if file.endswith(".csv"):
        dict_Par[file] = pd.read_csv(os.path.join(DirName, Folder_A, CaseName_Base, '2.Par', file))

# saving the dataframes in the new folders with a new name
for folder in FoldersToPaste:
    for case in CasesToPaste:
        for file in dict_Set:
            dict_Set[file].fillna("", inplace=True)
            dict_Set[file].to_csv(os.path.join(DirName, folder, case, '1.Set', file.split('_', 3)[0]+'_'+file.split('_', 3)[1]+'_'+file.split('_', 3)[2]+'_'+case+'.csv'), index=False)
        for file in dict_Par:
            dict_Set[file].fillna("", inplace=True)
            dict_Set[file].to_csv(os.path.join(DirName, folder, case, '2.Par', file.split('_', 3)[0]+'_'+file.split('_', 3)[1]+'_'+file.split('_', 3)[2]+'_'+case+'.csv'), index=False)
        path_to_ComputationTime_file = os.path.join(DirName, folder, case, '3.Out', 'ComputationTime.txt')
        with open(path_to_ComputationTime_file, 'w') as f:
            f.write(str(0.0))

# # reading the data
# df_Demand = pd.read_csv(os.path.join(DirName, Folder_A, CaseName_Base, '2.Par', 'oT_Data_Demand_' + CaseName_Base + '.csv'), index_col=[0,1,2])
# df_Demand.fillna("", inplace=True)
# # renaming the dataframe and saving it
# df_Demand.index.name = None
# print(os.path.join(DirName, Folder_D, CaseName_ByStage, '2.Par'))