import pandas as pd
import os
import glob
import numpy as np

def list_executions_from_Starnet_results(folder, case):
    filenames = glob.glob(pathname="oT_Input*",root_dir=folder)
    executions = [fn.split(case)[1][1:-4] for fn in filenames]
    return executions


def list_investments_candidates_from_execs(executions):
    l = [execution.strip("12345").strip("_cac") for execution in executions if
         execution != "Network_Existing_Generation_Full"]

    #     return list(set(l))
    return np.unique(l)



# folder_read = "../06.Benchmarking/C.The_1st_week_per_month/3-bus/3.Out/0.WoParallel"
# folder_read = "../06.Benchmarking/E.Representative_days_based_on_Line_Benefits_OptModel/3-bus/3.Out/0.WoParallel"

case = "9n"
folder_read = "../06.Benchmarking/C.The_1st_week_per_month/9n/3.Out/0.WoParallel"
folder_read = "../06.Benchmarking/E.Representative_days_based_on_Line_Benefits_OptModel/9n/3.Out/0.WoParallel"
folder_write = "9n_AC_12w_ext_o_dummy_LCOE"
folder_write = "9n_AC_fy_ext_o_dummy_LCOE"
input_export = True
output_export = True
executions = list_executions_from_Starnet_results(folder_read, case)
ics = list_investments_candidates_from_execs(executions)

per = 2030
sc = "sc01"

# Input stuff

if input_export:
    for exe in executions:

        # exe = executions[0]

        inp = pd.read_csv(f"{folder_read}/oT_Input_Data_{case}_{exe}.csv")

        generation_types = pd.read_csv("../Data/Samples_RTS24_ACOPF/oT_Data_Generation_Technology_RTS24.csv")
        techs_kept = ["Hydro", "Solar", "Wind"]
        variables = inp.Variable.unique()
        variables_to_drop_i = np.unique(generation_types[~generation_types.Technology.isin(techs_kept)].Unit)
        variables_to_keep_i = [v for v in variables if v not in (variables_to_drop_i)]

        # Filter the main frame on desired parameters:
        f_i_datasets = ~inp.Dataset.str.startswith('Matrix')
        f_i_tech = (inp.Variable.isin(variables_to_keep_i))
        f_i = (f_i_datasets) & (f_i_tech)
        # input_f = pd.DataFrame(all_input[f_i])
        inp_f_p = inp[f_i].pivot(index="LoadLevel", columns=["Variable"], values="Value")
        for ic in ics:
            if ic == exe.strip("12345").strip("_cac"):
                inp_f_p[ic] = exe[-1]
            else:
                inp_f_p[ic] = 0
        inp_f_p.to_csv(f"../Data/{folder_write}/input_f_{sc}_{exe}_{per}.csv")

#Output stuff

if output_export:
    for exe in executions:
        # exe = executions[0]

        # Read the output file into dataframe for a given execution
        outp = pd.read_csv(f"{folder_read}/oT_Output_Data_{case}_{exe}.csv")

        # loop over output types
        for key in ["SystemCosts", "PowerOutput", "PowerFlow"]:

            # Select one of the output types  and create a filter for the full frame
            #     key = outp_types[1]
            f_o = (outp.Dataset == key)

            if key == "SystemCosts":
                variables_to_keep_o = ["vTotalCCost", "vTotalECost", "vTotalGCost", "vTotalRCost"]
                print(sum(f_o))
                f_o = f_o & (outp.Variable.isin(variables_to_keep_o))
                print(sum(f_o))

            # Assign the filtered frame to a new frame
            outp_f = outp[f_o]
            # Pivot the filtered frame to have one row per timestamp, and one column per data type
            outp_f_p = outp_f.pivot(index="LoadLevel", columns=["Variable"], values="Value")
            # Store result in a csv file
            outp_f_p.to_csv(f"../Data/{folder_write}/output_f_{sc}_{exe}_{per}_{key}.csv")