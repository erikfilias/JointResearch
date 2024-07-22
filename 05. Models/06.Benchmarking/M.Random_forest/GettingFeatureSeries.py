import os
import time
import pandas as pd
from   sklearn.preprocessing import StandardScaler

InitialTime = time.time()

CaseName = 'RTS24_mod1'

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

path  = os.getcwd()
print("Current Directory:     ", path)
parent_path1 = os.path.abspath(os.path.join(path, os.pardir))
print("Parent Directory:      ", parent_path1)
parent_path2 = os.path.abspath(os.path.join(parent_path1, os.pardir))
print("Grand Parent Directory:", parent_path2)

# reading the Data_Demand
Data_Demand      = pd.read_csv(os.path.join(parent_path1, Folder_A, CaseName, '2.Par', f'oT_Data_Demand_{CaseName}.csv'), index_col=[0,1,2])
Data_VarGen      = pd.read_csv(os.path.join(parent_path1, Folder_A, CaseName, '2.Par', f'oT_Data_VariableMaxGeneration_{CaseName}.csv'), index_col=[0,1,2])
Data_OperCost    = pd.read_csv(os.path.join(parent_path1, Folder_B, CaseName, '3.Out', f'oT_Result_GenerationCost_{CaseName}.csv'), index_col=[0,1,2])
Data_LineBenefit = pd.read_csv(os.path.join(parent_path1, Folder_E, CaseName, '3.Out', f'oT_LineBenefit_Data_{CaseName}.csv'), index_col=[0])

# Data_Features    = pd.read_csv(os.path.join(parent_path2, '07.Interpretable_ML', 'FeatureImportances', f'{CaseName}.csv'), index_col=[0])
Data_Features    = pd.read_csv(os.path.join(parent_path2, '07.Interpretable_ML', 'FeatureImportances', f'{CaseName}_NO_OPC.csv'), index_col=[0])

print('Reading the csv files:                                        Done, ', round(time.time()-InitialTime,2), ' seconds')
StatTime = time.time()

# fill the NaN values with 0
Data_Demand.fillna(0, inplace=True)
Data_VarGen.fillna(0, inplace=True)
Data_OperCost.fillna(0, inplace=True)
Data_LineBenefit.fillna(0, inplace=True)

print('Filling the NaN values:                                       Done, ', round(time.time()-StatTime,2), ' seconds')
StatTime = time.time()

# Normalizing of each column of each dataframe considering their maximum value and minimum value
def normalize_dataframe(df):
    """
    Normalize each column of a dataframe using the min-max normalization.
    """
    return (df - df.min()) / (df.max() - df.min())

Data_Demand      = normalize_dataframe(Data_Demand)
Data_VarGen      = normalize_dataframe(Data_VarGen)
Data_OperCost    = normalize_dataframe(Data_OperCost)
Data_LineBenefit = normalize_dataframe(Data_LineBenefit)
# scaler = StandardScaler()
# Data_Demand[Data_Demand.columns] = scaler.fit_transform(Data_Demand)
# Data_VarGen[Data_VarGen.columns] = scaler.fit_transform(Data_VarGen)
# Data_OperCost[Data_OperCost.columns] = scaler.fit_transform(Data_OperCost)
# Data_LineBenefit[Data_LineBenefit.columns] = scaler.fit_transform(Data_LineBenefit)

print('Normalizing the dataframes:                                   Done, ', round(time.time()-StatTime,2), ' seconds')
StatTime = time.time()

Data_LineBenefit.index = Data_Demand.index

# merge the dataframes
Data = pd.concat([Data_Demand, Data_VarGen, Data_OperCost, Data_LineBenefit], axis=1)
Data.fillna(0, inplace=True)

print('Merging the dataframes:                                       Done, ', round(time.time()-StatTime,2), ' seconds')
StatTime = time.time()

# sort the Data_Features dataframe by the values of the column 'Importance'
Data_Features.sort_values(by='Importance', ascending=False, inplace=True)

# add new column 'Accumulated' to the Data_Features dataframe
Data_Features['Accumulated'] = Data_Features['Importance'].cumsum()

# select the features with 'Accumulated' less than 0.95
Data_Features_95 = Data_Features[Data_Features['Accumulated'] < 0.95]

# select the columns of the Data dataframe with the features selected
Data_reduced = Data[Data_Features_95['Name']]

# from the three indexes of the Data_reduced dataframe, select the third one as index
Data_reduced.index = Data_reduced.index.get_level_values(2)

# multiply each column of the Data_reduced dataframe by the value of the column 'Importance' in the Data_Features_95 dataframe
for idx, column in enumerate(Data_reduced.columns):
    importance_value = Data_Features_95.loc[Data_Features_95['Name'] == column, 'Importance'].values[0]
    Data_reduced.loc[:, column] = Data_reduced[column].mul(importance_value)

# save the Data_reduced dataframe in a csv file

Data_reduced.to_csv(os.path.join(parent_path1, Folder_M, CaseName, '3.Out', f'oT_FeatureSeries_{CaseName}.csv'), index=True)

print('Selecting the features and saving the csv file:               Done, ', round(time.time()-StatTime,2), ' seconds')