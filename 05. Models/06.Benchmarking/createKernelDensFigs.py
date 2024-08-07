import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def get_utils_FYMILP(case):
    df_utils_FYMILP = pd.read_csv(f"A.The_full_year_MILP/{case}/4.OutWoInv/oT_Result_NetworkUtilizationPerNode_DC_{case}.csv")
    #Only considering existing lines:
    df_utils_existing = df_utils_FYMILP[df_utils_FYMILP["Circuit"] == "eac1"]
    return df_utils_existing

def extract_FYMILP_one_line(all_utils_FYMILP,node_pair):
    filter_this_line = (all_utils_FYMILP["InitialNode"] == node_pair[0]) & (all_utils_FYMILP["FinalNode"] == node_pair[1])
    utils_this_line_FYMILP = all_utils_FYMILP[filter_this_line]

    ts_this_line_FYMILP = utils_this_line_FYMILP.GWh.to_numpy()
    return ts_this_line_FYMILP

def get_utils_cm(cm,nbc, case):
    return pd.read_csv(f"Y.FYTS_from_ByStages/{case}/{cm}/NetworkUtilization_nc{nbc}.csv",header = [0,1],index_col=0)

def extract_CM_one_line(all_utils_CM,node_pair):
    ts_this_line_CM = all_utils_CM.loc[:,node_pair].to_numpy()
    return ts_this_line_CM

def extract_unique_node_pairs(all_utils_FYMILP):
    unique_node_pairs = [tuple(x) for x in
                         all_utils_FYMILP[['InitialNode', 'FinalNode']].drop_duplicates().values.tolist()]
    return unique_node_pairs

def create_array_all_utils(unique_node_pairs,all_utils_FYMILP,all_utils_cm):
    # Initialize arrays for storing time series and KDE values for each method
    nb_lines = len(unique_node_pairs)
    ts_all_lines_CM = np.zeros(8736 * nb_lines)
    ts_all_lines_FYMILP = np.zeros(8736 * nb_lines)

    for i, node_pair in enumerate(unique_node_pairs):
        ts_this_line_FYMILP = extract_FYMILP_one_line(all_utils_FYMILP, node_pair)
        ts_this_line_CM = extract_CM_one_line(all_utils_cm,node_pair)

        i_start = i * 8736
        i_end = (i + 1) * 8736
        ts_all_lines_FYMILP[i_start:i_end] = ts_this_line_FYMILP
        ts_all_lines_CM[i_start:i_end] = ts_this_line_CM
    return ts_all_lines_FYMILP,ts_all_lines_CM

def get_filter(ts_all_lines_CM,ts_all_lines_FYMILP,min_value,max_value,filter_type):
    # Filter values based on thresholds
    if filter_type == "FYMILP":
        filter_greater_than = (ts_all_lines_FYMILP >= min_value) & (ts_all_lines_FYMILP <= max_value)
    elif filter_type =="Both":
        filter_greater_than = ((ts_all_lines_FYMILP >= min_value) & (ts_all_lines_FYMILP <= max_value)) & ((ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value))
    elif filter_type =="Either":
        filter_greater_than = ((ts_all_lines_FYMILP >= min_value) & (ts_all_lines_FYMILP <= max_value)) | ((ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value))
    elif filter_type =="CM":
        filter_greater_than = (ts_all_lines_CM >= min_value) & (ts_all_lines_CM <= max_value)
    elif filter_type == "None":
        filter_greater_than = (ts_all_lines_CM > -1) & (ts_all_lines_CM < 2)
    else:
        ValueError("Filter type not defined")

    return filter_greater_than

# Define the clustering methods and other constants
case = "RTS24_mod1"
clustering_methods = ["CHI", "R&D", "OPC", "HI", "OPT_LB"]  # Add the methods you want to analyze
nbc = 200
min_value =0.85
max_value = 1
filter_type = "Either"
xmin, xmax = min_value, max_value
ymin, ymax = min_value, max_value


# Prepare subplots
fig, axes = plt.subplots(len(clustering_methods), 1, figsize=(8, 16), sharex=True)
kde_values_list = []
# Prepare the grid for KDE
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])

for idx, cm in enumerate(clustering_methods):
    print(f"Creating timeseries arrays for {cm}")

    #Get all utilizations from saved files
    all_utils_FYMILP  =get_utils_FYMILP(case)
    all_utils_CM  = get_utils_cm(cm,nbc,case)

    #Extract unique node pairs of existing lines
    unique_node_pairs = extract_unique_node_pairs(all_utils_FYMILP)
    print(f"Considering {len(unique_node_pairs)} lines ")

    # Extract time series data for each node pair and clustering method
    ts_all_lines_FYMILP,ts_all_lines_CM = create_array_all_utils(unique_node_pairs,all_utils_FYMILP,all_utils_CM)
    # print(len(ts_all_lines_CM))
    # print(len(ts_all_lines_FYMILP))


    # Filter values based on thresholds
    filter_greater_than = get_filter(ts_all_lines_CM,ts_all_lines_FYMILP,min_value,max_value,filter_type)

    x = ts_all_lines_CM[filter_greater_than]
    y = ts_all_lines_FYMILP[filter_greater_than]
    #x = ts_all_lines_FYMILP[filter_greater_than]
    # x = ts_all_lines_FYMILP[filter_greater_than] + np.random.uniform(low=0.0, high=0.03, size=(len(y),))

    # Perform the kernel density estimate
    print("Performing kernel density estimate")
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    kde_values_list.append(f)

# Find the global max of KDE values for normalization
global_max_kde = max(np.max(f) for f in kde_values_list)

print("Plotting")
# Plot each KDE with consistent contour levels
for idx, (f, cm) in enumerate(zip(kde_values_list, clustering_methods)):
    ax = axes[idx]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Normalize the KDE values to range between 0 and 1
    f_normalized = f / global_max_kde

    # Contourf plot with normalized levels
    cfset = ax.contourf(xx, yy, f_normalized, cmap='Blues', levels=np.linspace(0, 1, 21))
    cset = ax.contour(xx, yy, f_normalized, colors='k', levels=np.linspace(0, 1, 11))
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_title(f'{cm}')
    ax.set_xlabel('CM')
    ax.set_ylabel('FYMILP')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(f"Z.Figures/KDE2D/Util/LineUtil_nc{nbc}_th{min_value}_{max_value}_ft{filter_type}.png")


