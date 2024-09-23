import pandas as pd
import numpy as np
import os
import argparse


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
    destination_folder = f"../Y.FYTS_from_ByStages/{CaseName_Base}/{cm}"
    filename = f"NetworkUtilization_nc{nbc}.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_NetworkUtilizationPerNode_DC_{CaseName_Base}_ByStages_nc{nbc}.csv")

    # Ensure data types are consistent
    df_duration["LoadLevel"] = df_duration["LoadLevel"].astype(str)
    df_ts_bs["LoadLevel"] = df_ts_bs["LoadLevel"].astype(str)
    df_ts_bs["InitialNode"] = df_ts_bs["InitialNode"].astype(str)
    df_ts_bs["FinalNode"] = df_ts_bs["FinalNode"].astype(str)

    # Filter out line candidates
    df_ts_bs = df_ts_bs[df_ts_bs["Circuit"] == "eac1"]

    # Merge df_ts_bs with df_duration based on LoadLevel
    df_merged = pd.merge(df_ts_bs, df_duration[['LoadLevel', 'Stage']], on='LoadLevel')

    # Create a MultiIndex from the unique node pairs
    unique_node_pairs = df_merged[['InitialNode', 'FinalNode']].drop_duplicates().values.tolist()
    multi_index = pd.MultiIndex.from_tuples(unique_node_pairs, names=['InitialNode', 'FinalNode'])

    # Initialize the DataFrame with LoadLevel as index and unique node pairs as columns
    frame_values = pd.DataFrame(index=df_duration['LoadLevel'], columns=multi_index)

    # Group by InitialNode, FinalNode, and LoadLevel to optimize the lookup
    grouped_df = df_merged.groupby(['InitialNode', 'FinalNode', 'LoadLevel'])

    for node_pair in unique_node_pairs:
        initial_node, final_node = node_pair

        # Initialize the full-year time series array for the node pair
        fy_ts = np.zeros(len(df_duration))

        # Iterate over LoadLevels
        for i, load_level in enumerate(df_duration['LoadLevel']):
            # Get the stage for this load level
            this_loadlevel_stage = df_duration.loc[df_duration['LoadLevel'] == load_level, 'Stage'].values[0]

            # Filter df_duration to find reduced load level (if exists)
            filtered_duration = df_duration[(df_duration['Stage'] == this_loadlevel_stage) &
                                            (df_duration['Duration'] == 1)]
            if not filtered_duration.empty:
                reduced_temp_load_level = filtered_duration['LoadLevel'].iloc[0]

                # Use pre-grouped data to find the corresponding value in df_ts_bs
                group = grouped_df.get_group((initial_node, final_node, str(reduced_temp_load_level)))

                if not group.empty:
                    fy_ts[i] = group['GWh'].values[0]

        # Assign the time series to the frame_values DataFrame
        frame_values[(initial_node, final_node)] = fy_ts

    # # Check if the destination folder exists, if not, create it
    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)
    #
    # # Write the dataframe to the destination folder
    # frame_values.to_csv(os.path.join(destination_folder, filename))

    return frame_values, destination_folder, filename


def create_curtailment_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"../Y.FYTS_from_ByStages_2/{CaseName_Base}/{cm}"

    filename = f"Curtailment_nc{nbc}.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_RESCurtailment_{CaseName_Base}_ByStages_nc{nbc}.csv")

    # Ensure data types are consistent: convert load levels to strings
    df_duration["LoadLevel"] = df_duration["LoadLevel"].astype(str)
    df_ts_bs["LoadLevel"] = df_ts_bs["LoadLevel"].astype(str)
    df_ts_bs["Unit"] = df_ts_bs["Unit"].astype(str)

    # Create mapping of load level to stage name
    load_level_stage_map = df_duration.set_index("LoadLevel")["Stage"].to_dict()
    all_load_levels = df_duration.LoadLevel

    # Create dataframe that that wil hold all timeseries
    frame_values = pd.DataFrame({"LoadLevel": all_load_levels})

    # Select a unit
    units = np.unique(df_ts_bs.Unit)

    for unit in units:

        # Initialize the full-year time series array
        fy_ts = np.zeros(len(all_load_levels))

        # Create full-year time series based on mapping and values of representative load levels
        for i, load_level in enumerate(all_load_levels):
            this_loadlevel_stage = load_level_stage_map[load_level]

            # Filter to find the correct reduced load level
            filtered_duration = df_duration[
                (df_duration["Stage"] == this_loadlevel_stage) & (df_duration["Duration"] == 1)]

            if not filtered_duration.empty:
                reduced_temp_load_level = filtered_duration.LoadLevel.iloc[0]

                # Find the corresponding value in df_ts_bs
                reduced_temp_value = df_ts_bs[
                    (df_ts_bs["Unit"] == unit) & (df_ts_bs["LoadLevel"] == str(reduced_temp_load_level))]

                if not reduced_temp_value.empty:
                    # Assuming you want to assign a value from reduced_temp_value to fy_ts
                    # You might need to aggregate if there are multiple values
                    fy_ts[i] = reduced_temp_value['MW'].iloc[0]  # Replace 'YourValueColumn' with the actual column name
        frame_values[unit] = fy_ts
    return frame_values, destination_folder, filename

def create_flow_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"../Y.FYTS_from_ByStages_2/{CaseName_Base}/{cm}"

    filename = f"Flow_per_node_nc{nbc}.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_NetworkFlowPerNode_{CaseName_Base}_ByStages_nc{nbc}.csv")

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
        print(node_pair)

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

def create_generation_cost_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"../Y.FYTS_from_ByStages_2/{CaseName_Base}/{cm}"

    filename = f"Generation_cost_nc{nbc}.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_GenerationCost_{CaseName_Base}_ByStages_nc{nbc}.csv")

    # Ensure data types are consistent: convert load levels to strings
    df_duration["LoadLevel"] = df_duration["LoadLevel"].astype(str)
    df_ts_bs["LoadLevel"] = df_ts_bs["LoadLevel"].astype(str)
    df_ts_bs["Unit"] = df_ts_bs["Unit"].astype(str)

    # Create mapping of load level to stage name
    load_level_stage_map = df_duration.set_index("LoadLevel")["Stage"].to_dict()
    all_load_levels = df_duration.LoadLevel

    # Create dataframe that that wil hold all timeseries
    frame_values = pd.DataFrame({"LoadLevel": all_load_levels})

    # Select a unit
    units = np.unique(df_ts_bs.Unit)

    for unit in units:

        # Initialize the full-year time series array
        fy_ts = np.zeros(len(all_load_levels))

        # Create full-year time series based on mapping and values of representative load levels
        for i, load_level in enumerate(all_load_levels):
            this_loadlevel_stage = load_level_stage_map[load_level]

            # Filter to find the correct reduced load level
            filtered_duration = df_duration[
                (df_duration["Stage"] == this_loadlevel_stage) & (df_duration["Duration"] == 1)]

            if not filtered_duration.empty:
                reduced_temp_load_level = filtered_duration.LoadLevel.iloc[0]

                # Find the corresponding value in df_ts_bs
                reduced_temp_value = df_ts_bs[
                    (df_ts_bs["Unit"] == unit) & (df_ts_bs["LoadLevel"] == str(reduced_temp_load_level))]

                if not reduced_temp_value.empty:
                    # Assuming you want to assign a value from reduced_temp_value to fy_ts
                    # You might need to aggregate if there are multiple values
                    fy_ts[i] = reduced_temp_value['MW'].iloc[0]  # Replace 'YourValueColumn' with the actual column name
        frame_values[unit] = fy_ts
    return frame_values, destination_folder, filename



def loop_over_nbcs(CaseName_Base, cm):
    for nbc in nbcs:
        print(nbc)
        if type == "util":
            frame_values, destination_folder, filename = create_utilization_fyts_frame(CaseName_Base, cm, nbc)
        elif type == "curt":
            frame_values, destination_folder, filename = create_curtailment_fyts_frame(CaseName_Base, cm, nbc)
        elif type == "flow":
            frame_values, destination_folder, filename = create_flow_fyts_frame(CaseName_Base, cm, nbc)
        else:
            print(f"Type: {type} undefined ")

        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        # And write dataframe to destination
        frame_values.to_csv(os.path.join(destination_folder, filename+"_2"))

#Define case
CaseName_Base = 'RTS24_mod1'
type = "util"
cm = "CHI"

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default="9n_mod1")
parser.add_argument('--type',    type=str, default="util")
parser.add_argument('--cm', type=str, default="CHI")
args = parser.parse_args()

CaseName_Base = args.case
type = args.type
cm = args.cm



nbcs = [10,20,50,100,200]

nbcs = [10,20,30,40,50,60,70,80,90,110,120,130,140,150]
nbcs = [10,30,50,70]#,90,110,130,150]
if cm == "All":
    #for cm_ in ["R&D", "OPT_LB", "CHI","HI","OPC"]:
    for cm_ in ["R&D","OPT_LB", "CHI", "HI", "OPC"]:

        folder_name = category_dict[cm_]
        loop_over_nbcs(CaseName_Base, cm_)
else:
    folder_name = category_dict[cm]
    loop_over_nbcs(CaseName_Base, cm)

