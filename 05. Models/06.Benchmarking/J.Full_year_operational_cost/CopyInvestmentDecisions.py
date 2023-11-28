import pandas as pd
import argparse
import re

#Specify the case, folder, and number of clusters
#TODO: As parser arguments

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--folder', type=str, default=None, help='Folder defining the clustering type')
parser.add_argument('--subfolder', type=str, default=None, help='Case')

args = parser.parse_args()

folder = args.folder
subfolder = args.subfolder

# Define a regular expression pattern to extract case and nb_stages
pattern = re.compile(r'(?P<case>.*?)_ByStages_nc(?P<nb_stages>.*)')

# Use the pattern to match the subfolder string
match = pattern.match(subfolder)

# Extract values from the match
if match:
    case = match.group('case')
    nb_stages = match.group('nb_stages')
    print(f"Extracted case: {case}, nb_stages: {nb_stages}")
else:
    print("No match found.")

# case = "3-bus"
# nb_stages = 10
# folder = "D.Representative_days_based_on_RES_and_Demand"

#First read the relevant investment results from the ByStages run
origin_folder = f"../{folder}/{case}_ByStages_nc{nb_stages}/3.Out/oT_Result_NetworkInvestment_{case}_ByStages_nc{nb_stages}.csv"
df_investment_results = pd.read_csv(origin_folder)

print(f"Importing investment results from {origin_folder}")
# Then read the network parameters file that will be adjusted to perform the operational run

# Specify the positions of the columns to be used as the index
index_columns = [0, 1, 2]

# Read the CSV file and set the specified columns as the index
df_parameters = pd.read_csv(f"{case}/2.Par/oT_Data_Network_{case}.csv", index_col=index_columns)

# Specify the names for the index columns, to have a coherent index with the network investment results
index_names = ['InitialNode', 'FinalNode', 'Circuit']

# Set the names for the index columns
df_parameters.index.set_names(index_names, inplace=True)
df_investment_results.set_index(index_names, inplace=True)

#Add a sensitivity column
df_investment_results["Sensitivity"] = "Yes"

# Change the relevant columns in the parameter dataframe based on imported investment results from the ByStages run
df_parameters["InvestmentFixed"] = df_investment_results["p.u."]
df_parameters["Sensitivity"] = df_investment_results["Sensitivity"]

#Finally, save the parameter file back to the original location, after altering the index names again to their original empty values
df_parameters.index.names = [None] * len(df_parameters.index.names)

destination_file= f"{case}/2.Par/oT_Data_Network_{case}.csv"
print(f"Pasting investment results in {destination_file}")
df_parameters.to_csv(destination_file)