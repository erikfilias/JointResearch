import pandas as pd
import numpy as np
###############
# Cost
###############

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

def get_cost_diffs_FYMILP(df_cost_FY,cm,nbc,case):
    df_cost_CM = pd.read_csv(f"J.Full_year_operational_cost/Results/{category_dict[cm]}/{case}_ByStages_nc{nbc}/3.Out/oT_Result_CostSummary_{case}_ByStages_nc{nbc}.csv")
    return (df_cost_CM.set_index("Cost/Payment")["MEUR"] - df_cost_FY.set_index("Cost/Payment")["MEUR"])

def get_cost_diffs_FYWOI(df_cost_FY,cm,nbc,case):
    df_cost_CM = pd.read_csv(f"{category_dict[cm]}/{case}_ByStages_nc{nbc}/4.OutWoInv/oT_Result_CostSummary_{case}_ByStages_nc{nbc}.csv")
    return (df_cost_CM.set_index("Cost/Payment")["MEUR"] - df_cost_FY.set_index("Cost/Payment")["MEUR"])
# def get_cost_WOI(df_cost_FY,cm,nbc,case):
#     df_cost_CM = pd.read_csv(f"J.Full_year_operational_cost/Results/{category_dict[cm]}/{case}_ByStages_nc{nbc}/3.Out/oT_Result_CostSummary_{case}_ByStages_nc{nbc}.csv")
#     return (df_cost_CM.set_index("Cost/Payment")["MEUR"] - df_cost_FY.set_index("Cost/Payment")["MEUR"])
#
def get_total_cost_diff_with_FYFixedInv(df_cost_FY,cm,nbc,case):
    df_cost_CM = pd.read_csv(f"J.Full_year_operational_cost/Results/{category_dict[cm]}/{case}_ByStages_nc{nbc}/3.Out/oT_Result_CostSummary_{case}_ByStages_nc{nbc}.csv")
    return (df_cost_CM.set_index("Cost/Payment")["MEUR"] - df_cost_FY.set_index("Cost/Payment")["MEUR"]).iloc[0]

#############
### Utils
#############
def get_utils_FYWOI(case):
    df_utils_FYWOI = pd.read_csv(f"A.The_full_year_MILP/{case}/4.OutWoInv/oT_Result_NetworkUtilizationPerNode_DC_{case}.csv")
    # Only considering existing lines:
    df_utils_existing = df_utils_FYWOI[df_utils_FYWOI["Circuit"] == "eac1"]
    return df_utils_existing

def get_utils_cm(cm, nbc, case):
    return pd.read_csv(f"Y.FYTS_from_ByStages/{case}/{cm}/NetworkUtilization_nc{nbc}_2.csv", header=[0, 1], index_col=0)

def get_filter(ts_all_lines_CM, ts_all_lines_FYWOI, min_value, max_value, filter_type):
    # Filter values based on thresholds
    if filter_type == "FYWOI":
        filter_greater_than = (ts_all_lines_FYWOI >= min_value) & (ts_all_lines_FYWOI <= max_value)
    elif filter_type == "Both":
        filter_greater_than = ((ts_all_lines_FYWOI >= min_value) & (ts_all_lines_FYWOI <= max_value)) & ((ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value))
    elif filter_type == "Either":
        filter_greater_than = ((ts_all_lines_FYWOI >= min_value) & (ts_all_lines_FYWOI <= max_value)) | ((ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value))
    elif filter_type == "CM":
        filter_greater_than = (ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value)
    elif filter_type == "None":
        filter_greater_than = (ts_all_lines_CM > -1) & (ts_all_lines_CM < 2)
    else:
        raise ValueError("Filter type not defined")

    return filter_greater_than


def get_util_diffs(all_utils_FYWOI,cm,nbc,case):
    min_value = 0.85
    max_value = 1

    filter_type = "Either"
    # Get all utilizations from saved files

    all_utils_CM = get_utils_cm(cm, nbc, case)

    # Extract unique node pairs of existing lines
    unique_node_pairs = extract_unique_node_pairs(all_utils_FYWOI)
    print(f"Considering {len(unique_node_pairs)} lines ")

    # Extract time series data for each node pair and clustering method
    ts_all_lines_FYWOI, ts_all_lines_CM = create_array_all_values(unique_node_pairs, all_utils_FYWOI, all_utils_CM)

    # Filter values based on thresholds
    filter_greater_than = get_filter(ts_all_lines_CM, ts_all_lines_FYWOI, min_value, max_value, filter_type)

    x = ts_all_lines_CM[filter_greater_than]
    y = ts_all_lines_FYWOI[filter_greater_than]

    return x - y


###################
#Line flows
##################
def get_line_flows_FYWOI(case):
    df_line_flows_FYWOI = pd.read_csv(f"A.The_full_year_MILP/{case}/4.OutWoInv/oT_Result_NetworkFlowPerNode_{case}.csv")
    # Only considering existing lines:
    df_line_flows_existing = df_line_flows_FYWOI[df_line_flows_FYWOI["Circuit"] == "eac1"]
    return df_line_flows_existing

def get_line_flows_cm(cm, nbc, case):
    return pd.read_csv(f"Y.FYTS_from_ByStages/{case}/{cm}/Flow_per_node_nc{nbc}_2.csv", header=[0, 1], index_col=0)

def get_line_flow_arrays(all_line_flows_FYWOI,cm,nbc,case):
    # Get all utilizations from saved files
    all_line_flows_CM = get_line_flows_cm(cm, nbc, case)

    # Extract unique node pairs of existing lines
    unique_node_pairs = extract_unique_node_pairs(all_line_flows_FYWOI)
    print(f"Considering {len(unique_node_pairs)} lines ")

    # Extract time series data for each node pair and clustering method
    ts_all_lines_FYWOI, ts_all_lines_CM = create_array_all_values(unique_node_pairs, all_line_flows_FYWOI,
                                                                      all_line_flows_CM)

    x = ts_all_lines_CM
    y = ts_all_lines_FYWOI

    return x,y

############################
#Line flows & utils generric
############################
def extract_value_FYWOI_one_line(all_values_FYWOI, node_pair):
    filter_this_line = (all_values_FYWOI["InitialNode"] == node_pair[0]) & (all_values_FYWOI["FinalNode"] == node_pair[1])
    values_this_line_FYWOI = all_values_FYWOI[filter_this_line]

    ts_this_line_FYWOI = values_this_line_FYWOI.GWh.to_numpy()
    return ts_this_line_FYWOI

def extract_value_CM_one_line(all_values_CM, node_pair):
    ts_this_line_CM = all_values_CM.loc[:, node_pair].to_numpy()
    return ts_this_line_CM

def extract_unique_node_pairs(all_values_FYWOI):
    unique_node_pairs = [tuple(x) for x in all_values_FYWOI[['InitialNode', 'FinalNode']].drop_duplicates().values.tolist()]
    return sorted(unique_node_pairs)

def create_array_all_values(unique_node_pairs, all_values_FYWOI, all_values_cm):
    # Initialize arrays for storing time series and KDE values for each method
    nb_lines = len(unique_node_pairs)
    ts_all_lines_CM = np.zeros(8736 * nb_lines)
    ts_all_lines_FYWOI = np.zeros(8736 * nb_lines)

    for i, node_pair in enumerate(unique_node_pairs):
        ts_this_line_FYWOI = extract_value_FYWOI_one_line(all_values_FYWOI, node_pair)
        ts_this_line_CM = extract_value_CM_one_line(all_values_cm, node_pair)

        i_start = i * 8736
        i_end = (i + 1) * 8736
        ts_all_lines_FYWOI[i_start:i_end] = ts_this_line_FYWOI
        ts_all_lines_CM[i_start:i_end] = ts_this_line_CM
    return ts_all_lines_FYWOI, ts_all_lines_CM

###################
#Curtailment
##################
def get_curts_FYWOI(case):
    df_curts_FYWOI = pd.read_csv(f"A.The_full_year_MILP/{case}/4.OutWoInv/oT_Result_RESCurtailment_{case}.csv")
    # Only considering existing lines:
    # df_utils_existing = df_utils_FYWOI[df_utils_FYWOI["Circuit"] == "eac1"]

    return df_curts_FYWOI
def sum_curts_per_timestep_FYWOI(df_curts_per_unit):
    return df_curts_per_unit.pivot_table(values="MW", index="LoadLevel", aggfunc="sum")["MW"]

def get_curts_cm(cm, nbc, case):
    return pd.read_csv(f"Y.FYTS_from_ByStages/{case}/{cm}/Curtailment_nc{nbc}_2.csv", header=[0], index_col=0)
def sum_curts_per_timestep_cm(df_curts_per_unit):
    #return df_curts_per_unit.set_index("LoadLevel").sum(axis=1)
    return df_curts_per_unit.sum(axis=1)


###################
#Generation cost
##################

