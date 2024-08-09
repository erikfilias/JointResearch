import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Folder_A = 'A.The_full_year_MILP'
Folder_B = 'B.Operation_cost'
Folder_D = 'D.Representative_days_based_on_RES_and_Demand'
Folder_E = 'E.Representative_days_based_on_Line_Benefits_OptModel'
Folder_K = 'K.Investments_per_hour'
Folder_L = 'L.Cont_Investments_per_hour'

category_dict = { "FYMILP": Folder_A,
                 "OPC":Folder_B,
                 "R&D": Folder_D ,
                 "OPT_LB": Folder_E,
                 "HI": Folder_K,
                  "CHI": Folder_L,
                }


def create_utilization_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"Y.FYTS_from_ByStages/{CaseName_Base}/{cm}"

    filename = f"NetworkUtilization_nc{nbc}.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_NetworkUtilizationPerNode_DC_{CaseName_Base}_ByStages_nc{nbc}.csv")

    # Ensure data types are consistent: convert load levels to strings
    df_duration["LoadLevel"] = df_duration["LoadLevel"].astype(str)
    df_ts_bs["LoadLevel"] = df_ts_bs["LoadLevel"].astype(str)
    df_ts_bs["InitialNode"] = df_ts_bs["InitialNode"].astype(str)
    df_ts_bs["FinalNode"] = df_ts_bs["FinalNode"].astype(str)

    # Filter out line candidates
    df_ts_bs = df_ts_bs[df_ts_bs["Circuit"] == "eac1"]

    # Create mapping of load level to stage name
    load_level_stage_map = df_duration.set_index("LoadLevel")["Stage"].to_dict()
    all_load_levels = df_duration.LoadLevel

    # Create dataframe that that wil hold all timeseries
    frame_values = pd.DataFrame({"LoadLevel": all_load_levels})

    # Get unique node pairs connected by an existing line
    unique_node_pairs = [tuple(x) for x in df_ts_bs[['InitialNode', 'FinalNode']].drop_duplicates().values.tolist()]

    # Create a MultiIndex from the unique node pairs
    multi_index = pd.MultiIndex.from_tuples(unique_node_pairs, names=['InitialNode', 'FinalNode'])

    # Initialize a DataFrame to hold all time series with MultiIndex
    frame_values = pd.DataFrame(index=all_load_levels, columns=multi_index)

    # Select one node pair
    node_pair = unique_node_pairs[0]

    for node_pair in unique_node_pairs:
        #print(node_pair)

        initial_node, final_node = node_pair

        # Initialize the full-year time series array
        fy_ts = np.zeros(len(all_load_levels))
        for i, load_level in enumerate(all_load_levels):
            this_loadlevel_stage = load_level_stage_map[load_level]

            # Filter to find the correct reduced load level
            filtered_duration = df_duration[
                (df_duration["Stage"] == this_loadlevel_stage) & (df_duration["Duration"] == 1)]

            if not filtered_duration.empty:
                reduced_temp_load_level = filtered_duration.LoadLevel.iloc[0]

                # Find the corresponding value in df_ts_bs
                reduced_temp_value = df_ts_bs[
                    (df_ts_bs["InitialNode"] == node_pair[0]) & (df_ts_bs["FinalNode"] == node_pair[1]) & (
                                df_ts_bs["LoadLevel"] == str(reduced_temp_load_level))]

                if not reduced_temp_value.empty:
                    # Assuming you want to assign a value from reduced_temp_value to fy_ts
                    # You might need to aggregate if there are multiple values
                    fy_ts[i] = reduced_temp_value['GWh'].iloc[0]
        frame_values[(initial_node, final_node)] = fy_ts
    return frame_values, destination_folder, filename

#Define case
CaseName_Base = 'RTS24_mod1'


cm = "OPT_LB"
folder_name = category_dict[cm]

nbcs = [10,20,50,100,200]

for nbc in nbcs:
    print(nbc)
    frame_values,destination_folder,filename = create_utilization_fyts_frame(CaseName_Base,cm,nbc)
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    #And write dataframe to destination
    frame_values.to_csv(os.path.join(destination_folder,filename))