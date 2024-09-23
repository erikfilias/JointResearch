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

    filename = f"NetworkUtilization_nc{nbc}_2.csv"

    # Read input data
    df_duration = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/2.Par/oT_Data_Duration_{CaseName_Base}_ByStages_nc{nbc}.csv")
    df_ts_bs = pd.read_csv(
        f"../{folder_name}/{CaseName_Base}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_NetworkUtilizationPerNode_DC_{CaseName_Base}_ByStages_nc{nbc}.csv")

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

    #Initilize dictionaries that maps stage name to the representative load level
    stage_to_repr_load_level_map = df_duration[df_duration["Duration"] == 1].set_index("Stage")["LoadLevel"].to_dict()

    for node_pair in unique_node_pairs:
        #print(node_pair)

        initial_node, final_node = node_pair

        filter_node_pair = (df_ts_bs["InitialNode"] == initial_node) & (df_ts_bs["FinalNode"] == final_node)
        repr_load_level_to_value_map = df_ts_bs[filter_node_pair].set_index("LoadLevel")["GWh"].to_dict()

        frame_values[(initial_node, final_node)] = frame_values.index.map(load_level_stage_map).map(
            stage_to_repr_load_level_map).map(repr_load_level_to_value_map)

    return frame_values, destination_folder, filename

def create_curtailment_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"../Y.FYTS_from_ByStages/{CaseName_Base}/{cm}"
    filename = f"Curtailment_nc{nbc}_2.csv"

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

    # Create a mapping of stage to the representative load level (where duration == 1)
    stage_to_repr_load_level_map = df_duration[df_duration["Duration"] == 1].set_index("Stage")["LoadLevel"].to_dict()

    # Create a mapping of load level and unit to MW value in df_ts_bs
    load_unit_to_value_map = df_ts_bs.set_index(["Unit", "LoadLevel"])["MW"].to_dict()

    # Create dataframe that will hold all timeseries, indexed by load levels
    all_load_levels = df_duration["LoadLevel"]
    frame_values = pd.DataFrame(index=all_load_levels)

    # Select unique units
    units = np.unique(df_ts_bs["Unit"])

    # Loop through units
    for unit in units:
        # Initialize the full-year time series array for this unit
        fy_ts = np.zeros(len(all_load_levels))

        # Map each load level to the corresponding value
        for i, load_level in enumerate(all_load_levels):
            this_loadlevel_stage = load_level_stage_map[load_level]

            # Find the representative load level for the current stage
            repr_load_level = stage_to_repr_load_level_map.get(this_loadlevel_stage)

            if repr_load_level:
                # Get the corresponding value for this unit and representative load level
                value = load_unit_to_value_map.get((unit, repr_load_level), 0)
                fy_ts[i] = value  # Assign the value to the full-year time series array

        # Add the full-year time series for the current unit to the DataFrame
        frame_values[unit] = fy_ts

    return frame_values, destination_folder, filename


def create_flow_fyts_frame(CaseName_Base, cm, nbc):
    destination_folder = f"../Y.FYTS_from_ByStages/{CaseName_Base}/{cm}"

    filename = f"Flow_per_node_nc{nbc}_2.csv"

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

    node_pair_data_map = {}
    stage_to_repr_load_level_map = df_duration[df_duration["Duration"] == 1].set_index("Stage")["LoadLevel"].to_dict()

    for node_pair in unique_node_pairs:
        print(node_pair)

        initial_node, final_node = node_pair
        filter_node_pair = (df_ts_bs["InitialNode"] == initial_node) & (df_ts_bs["FinalNode"] == final_node)
        node_pair_data_map[node_pair] = df_ts_bs[filter_node_pair].set_index("LoadLevel")["GWh"].to_dict()

        # Map the load levels to stages and then to representative load levels
        mapped_values = (
            frame_values.index
            .map(load_level_stage_map)
            .map(stage_to_repr_load_level_map)
            .map(node_pair_data_map[node_pair])
        )

        frame_values[(initial_node, final_node)] = mapped_values

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
        frame_values.to_csv(os.path.join(destination_folder, filename ))

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default="9n_mod1")
parser.add_argument('--type',    type=str, default="util")
parser.add_argument('--cm', type=str, default="CHI")
args = parser.parse_args()

CaseName_Base = args.case
type = args.type
cm = args.cm



# nbcs = [10,20,50,100,200]

nbcs = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
# nbcs = [10,30,50,70]#,90,110,130,150]
nbcs = [100]#,90,110,130,150]
if cm == "All":
    #for cm_ in ["R&D", "OPT_LB", "CHI","HI","OPC"]:
    for cm_ in ["R&D","OPT_LB", "CHI", "HI", "OPC"]:

        folder_name = category_dict[cm_]
        loop_over_nbcs(CaseName_Base, cm_)
else:
    folder_name = category_dict[cm]
    loop_over_nbcs(CaseName_Base, cm)

