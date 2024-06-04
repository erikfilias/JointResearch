import os
import time
import pandas as pd

InitialTime = time.time()

filter_TOOT = "oT_TOOT_Output_Data_"
filter_PINT = "oT_PINT_Output_Data_"

# DirName = os.path.dirname(__file__)
# DirName  = "C:\\Users\\ealvarezq\\Downloads\\LB_IEEE118_mod1\\IEEE118_mod1"
DirName  = "C:\\Users\\ealvarezq\\Downloads\\DCOPF"
CaseName = "9n_mod1"

df = pd.read_csv(os.path.join(DirName, f"oT_Result_NN_Output_{CaseName}.csv"))
print('Reading the file took', time.time() - InitialTime, 'seconds')

# filter vTotalGCost, vTotalECost, vTotalCCost, and vTotalRCost form the column variable
df = df.loc[df['Dataset'].isin(['SystemCosts'])]
df = df.loc[df['Variable'].isin(['vTotalGCost', 'vTotalECost', 'vTotalCCost', 'vTotalRCost'])]

print('Filtering the file took', time.time() - InitialTime, 'seconds')

# pivot the table
df = pd.pivot_table(df, values='Value', index=['LoadLevel', 'Execution'])
df = df.reset_index()

# Splitting df into based on the column values of execution
df_Existing = df.loc[df['Execution'].isin(['Network_Existing_Generation_Full'])]
df_Existing = df_Existing[['LoadLevel', 'Value']]
df_Existing = df_Existing.rename(columns={'Value': 'Existing'})
df_Existing.set_index('LoadLevel', inplace=True)
df_Full     = df.loc[df['Execution'].isin(['Network_Full_Generation_Full'])]
df_Full     = df_Full[['LoadLevel', 'Value']]
df_Full     = df_Full.rename(columns={'Value': 'Full'})
df_Full.set_index('LoadLevel', inplace=True)

# getting CSV files of the TOOT and PINT based on the filter_TOOT and filter_PINT
directories_TOOT = [f[len(filter_TOOT+CaseName+'_Network_Line_Out_'):len(f)-4] for f in os.listdir(DirName) if filter_TOOT in f]
directories_PINT = [f[len(filter_PINT+CaseName+'_Network_Line_In_'):len(f)-4] for f in os.listdir(DirName) if filter_PINT in f]

print('Getting the directories took', time.time() - InitialTime, 'seconds')

# getting the TOOT and PINT dataframes
df_TOOT = {}
df_PINT = {}
for directory in directories_TOOT:
    df_TOOT[directory] = pd.read_csv(os.path.join(DirName, filter_TOOT+CaseName+'_Network_Line_Out_'+directory+'.csv'))
    df_TOOT[directory] = df_TOOT[directory].loc[df_TOOT[directory]['Dataset'].isin(['SystemCosts'])]
    df_TOOT[directory] = df_TOOT[directory].loc[df_TOOT[directory]['Variable'].isin(['vTotalGCost', 'vTotalECost', 'vTotalCCost', 'vTotalRCost'])]
    df_TOOT[directory] = pd.pivot_table(df_TOOT[directory], values='Value', index=['LoadLevel', 'Execution'])
    df_TOOT[directory] = df_TOOT[directory].reset_index()
for directory in directories_PINT:
    df_PINT[directory] = pd.read_csv(os.path.join(DirName, filter_PINT+CaseName+'_Network_Line_In_'+directory+'.csv'))
    df_PINT[directory] = df_PINT[directory].loc[df_PINT[directory]['Dataset'].isin(['SystemCosts'])]
    df_PINT[directory] = df_PINT[directory].loc[df_PINT[directory]['Variable'].isin(['vTotalGCost', 'vTotalECost', 'vTotalCCost', 'vTotalRCost'])]
    df_PINT[directory] = pd.pivot_table(df_PINT[directory], values='Value', index=['LoadLevel', 'Execution'])
    df_PINT[directory] = df_PINT[directory].reset_index()

print('Getting the dataframes took', time.time() - InitialTime, 'seconds')

# merging the dataframes of TOOT and PINT in only one
df1_TOOT = pd.concat(df_TOOT)
df1_TOOT = df1_TOOT.reset_index()
df1_TOOT = pd.pivot_table(df1_TOOT, values='Value', index=['LoadLevel'], columns=['Execution'])

df1_PINT = pd.concat(df_PINT)
df1_PINT = df1_PINT.reset_index()
df1_PINT = pd.pivot_table(df1_PINT, values='Value', index=['LoadLevel'], columns=['Execution'])

print('Merging the dataframes took', time.time() - InitialTime, 'seconds')

# merging or concat the dataframes of TOOT with the PINT, df_Existing and df_Full
Output1 = pd.concat([df1_TOOT, df1_PINT, df_Existing, df_Full], axis=1)
Output1.to_csv(os.path.join(DirName, f"oT_OperatingCost_AllExecutions_{CaseName}.csv"), index=True)

print('Saving the CSV file related to the operating cost took', time.time() - InitialTime, 'seconds')

# Making the math operation to get the line benefit
for directory in directories_PINT:
    df1_PINT[f'Network_Line_In_{directory}'] =  df_Existing['Existing'] - df1_PINT[f'Network_Line_In_{directory}']

for directory in directories_TOOT:
    df1_TOOT[f'Network_Line_Out_{directory}'] =  df1_TOOT[f'Network_Line_Out_{directory}'] - df_Full['Full']

print('Making the math operation took', time.time() - InitialTime, 'seconds')

# mean of the PINT and TOOT to get the line benefit
Output2 = df1_TOOT.copy()
for directory in directories_PINT:
    Output2[f'Network_Line_Out_{directory}'] = (df1_PINT[f'Network_Line_In_{directory}'] + df1_TOOT[f'Network_Line_Out_{directory}']) / 2

Output2.to_csv(os.path.join(DirName, f"oT_LineBenefit_PerLine_{CaseName}.csv"), index=True)

print('Saving the CSV file related to the line benefit took', time.time() - InitialTime, 'seconds')