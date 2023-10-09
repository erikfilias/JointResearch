import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random

def list_executions(per,sc,folder):
    filenames = os.listdir(f"{folder}")
    executions = [fn.split(per)[0].split(sc)[1][1:-1] for fn in filenames]
    return np.unique(executions)

# def load_data(folder,executions,period,sc):
#     dfs_in = dict()
#     dfs_out = dict()
#     for execution in executions:
#         # Read the data from desired execution
#         df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0],index_col=0)
#         df_out_e = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}.csv", header=[0],index_col=0)
#
#         print(f"input_f_{sc}_{execution}_{period}.csv")
#
#         # And order the variables:
#
#         print(len(df_in_e.columns))
#         for col in df_out_e.columns:
#             df_out_e[col] = df_out_e[col].astype(float)
#         for col in df_in_e.columns:
#             df_in_e[col] = df_in_e[col].astype(float)
#         dfs_in[execution] = df_in_e
#         dfs_out[execution] = df_out_e
#     return dfs_in,dfs_out

def load_data_ext_out(folder, executions, period, sc, il_os=None,output = "SystemCosts",include_inv_dummies =True):
    dfs_in = dict()
    dfs_out = dict()
    dfs_inter = dict()
    for execution in executions:
        # Read the data from desired execution
        df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0], index_col=0)
        df_out_e = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}_{output}.csv", header=[0], index_col=0)



        print(f"input_f_{sc}_{execution}_{period}.csv")

        #Drop the clumns that are related to the investment dummy variables
        if not include_inv_dummies:
            df_in_e = df_in_e[df_in_e.columns.drop(list(df_in_e.filter(regex='Network_Line_In')))]


        # And order the variables:
        print(len(df_in_e.columns))
        for col in df_out_e.columns:
            df_out_e[col] = df_out_e[col].astype(float)
        for col in df_in_e.columns:
            df_in_e[col] = df_in_e[col].astype(float)

        if il_os != None:
            dfs_ilo = dict()
            for il_o in il_os:
                df_inter = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}_{il_o}.csv", header=[0],
                                       index_col=0)
                for col in df_inter.columns:
                    df_inter[col] = df_inter[col].astype(float)
                dfs_ilo[il_o] = df_inter
            dfs_inter[execution] = dfs_ilo

        dfs_in[execution] = df_in_e
        dfs_out[execution] = df_out_e

    return dfs_in, dfs_out, dfs_inter

# def load_data_input_only(folder, executions, period, sc):
#     dfs_in = dict()
#
#     for execution in executions:
#         # Read the data from desired execution
#         df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0], index_col=0)
#
#         print(f"input_f_{sc}_{execution}_{period}.csv")
#
#         print(len(df_in_e.columns))
#         for col in df_in_e.columns:
#             df_in_e[col] = df_in_e[col].astype(float)
#
#         dfs_in[execution] = df_in_e
#
#     return dfs_in

def join_frames_inter_layer(dfs_inter,executions):
    dfs_inter_j = dict()
    for execution in executions:
        dfs_inter_j[execution] = pd.concat([dfs_inter[execution][t] for t in sorted(dfs_inter[execution].keys())],axis=1)
    return dfs_inter_j

def trim_columns_to_common(dfs_inter_j):
    common_columns = list(set.intersection(*(set(df.columns) for df in dfs_inter_j.values())))
    # Filter DataFrames to keep only common columns
    filtered_dataframes_dict = {name: df[common_columns] for name, df in dfs_inter_j.items()}
    return filtered_dataframes_dict


# def split_tr_val_te(dfs_in,dfs_out,executions,te_s,val_s):
#     ts_in = dict()
#     ts_out = dict()
#
#     ts_in["train"] = dict()
#     ts_in["test"] = dict()
#     ts_in["val"] = dict()
#
#     ts_out["train"] = dict()
#     ts_out["test"] = dict()
#     ts_out["val"] = dict()
#
#     # Test size as fraction of full dataset, validation size as fraction of training data set
#     test_size, validation_size = te_s, val_s
#
#     for execution in executions:
#         # Convert input dataframes numpy arrays sum the columns of the output:
#         np_in = dfs_in[execution].to_numpy()
#         np_out = dfs_out[execution].to_numpy().sum(axis=1)
#
#         # We don't normalize the separate runs, but will do it afterward, all together
#
#         # Convert to torch tensors
#         t_in = torch.from_numpy(np_in)
#         t_out = torch.from_numpy(np_out)
#
#         # And split into train, validation, and test set:
#         train_in, ts_in["test"][execution], train_out, ts_out["test"][execution] = train_test_split(t_in, t_out,
#                                                                                                     test_size=test_size,
#                                                                                                     shuffle=False)
#         ts_in["train"][execution], ts_in["val"][execution], ts_out["train"][execution], ts_out["val"][
#             execution] = train_test_split(train_in, train_out, test_size=validation_size, shuffle=False)
#     return ts_in,ts_out

def concat_all_exec_fy(dfs_in, dfs_out, dfs_inter_j,executions,normalize_out = False):
    first = True
    for execution in executions:
        np_in = dfs_in[execution].to_numpy()
        np_out = dfs_out[execution].to_numpy().sum(axis=1)
        np_inter = dfs_inter_j[execution].to_numpy()

        t_in = torch.from_numpy(np_in)
        t_out = torch.from_numpy(np_out)
        t_inter = torch.from_numpy(np_inter)
        if first:
            t_in_fy = t_in
            t_out_fy = t_out
            t_inter_fy = t_inter
            first = False
        else:
            t_in_fy = torch.cat((t_in_fy,t_in))
            t_out_fy = torch.cat((t_out_fy, t_out))
            t_inter_fy = torch.cat((t_inter_fy,t_inter))
    maxs = dict()
    maxs["in"] = t_in_fy.abs().max(dim=0).values
    maxs["inter"] = t_inter_fy.abs().max(dim=0).values
    maxs["out"] = t_out_fy.abs().max(dim=0).values

    t_in_fy = torch.nan_to_num(t_in_fy / maxs["in"])
    t_inter_fy = torch.nan_to_num(t_inter_fy / maxs["inter"])
    if normalize_out:
        t_out_fy = torch.nan_to_num(t_out_fy / maxs["out"])

    return t_in_fy,t_out_fy,t_inter_fy,maxs

def split_tr_val_te_ext_out(dfs_in, dfs_out, dfs_inter_j, executions, te_s, val_s,shuffle = True):
    ts_in = dict()
    ts_out = dict()
    ts_inter = dict()

    ts_in["train"] = dict()
    ts_in["test"] = dict()
    ts_in["val"] = dict()

    ts_out["train"] = dict()
    ts_out["test"] = dict()
    ts_out["val"] = dict()

    ts_inter["train"] = dict()
    ts_inter["test"] = dict()
    ts_inter["val"] = dict()

    # Test size as fraction of full dataset, validation size as fraction of training data set
    test_size, validation_size = te_s, val_s

    for execution in executions:
        # Convert input dataframes numpy arrays sum the columns of the output:
        np_in = dfs_in[execution].to_numpy()
        np_out = dfs_out[execution].to_numpy().sum(axis=1)
        np_inter = dfs_inter_j[execution].to_numpy()

        if shuffle:
            seed = 0
            rng = np.random.default_rng(seed)
            np_in = rng.permutation(np_in,axis=0)

            rng = np.random.default_rng(seed)
            np_out = rng.permutation(np_out, axis=0)

            rng = np.random.default_rng(seed)
            np_inter = rng.permutation(np_inter, axis=0)


        # We don't normalize the separate runs, but will do it afterward, all together

        # Convert to torch tensors
        t_in = torch.from_numpy(np_in)
        t_out = torch.from_numpy(np_out)
        t_inter = torch.from_numpy(np_inter)

        # And split into train, validation, and test set:
        train_in, ts_in["test"][execution], train_out, ts_out["test"][execution], train_inter, ts_inter["test"][
            execution] = train_test_split(t_in, t_out, t_inter,
                                          test_size=test_size,
                                          shuffle=False)
        ts_in["train"][execution], ts_in["val"][execution], ts_out["train"][execution], ts_out["val"][
            execution], ts_inter["train"][execution], ts_inter["val"][
            execution] = train_test_split(train_in, train_out, train_inter, test_size=validation_size, shuffle=False)
    return ts_in, ts_out, ts_inter

# def split_tr_val_te_by_exec(dfs_in,dfs_out,executions,te_s,val_s,randomized = False):
#     ts_in = dict()
#     ts_out = dict()
#
#     ts_in["train"] = dict()
#     ts_in["test"] = dict()
#     ts_in["val"] = dict()
#
#     ts_out["train"] = dict()
#     ts_out["test"] = dict()
#     ts_out["val"] = dict()
#
#     nb_executions = len(dfs_in)
#
#     # Test size as fraction of full dataset, validation size as fraction of training data set
#     nb_executions = len(dfs_in)
#     nb_test = int(round(nb_executions * te_s))
#     nb_val = int(round((nb_executions - nb_test) * val_s))
#     nb_train = int(round((nb_executions - nb_test) * (1 - val_s)))
#     assert (nb_executions == nb_test + nb_train + nb_val)
#
#     l_keys = list(dfs_in)
#     if randomized:
#         random.shuffle(dfs_in)
#
#     for i, execution in enumerate(executions):
#         # Convert input dataframes numpy arrays sum the columns of the output:
#         np_in = dfs_in[execution].to_numpy()
#         np_out = dfs_out[execution].to_numpy().sum(axis=1)
#         # Convert to torch tensors
#         t_in = torch.from_numpy(np_in)
#         t_out = torch.from_numpy(np_out)
#
#         if i < nb_train:
#             ts_in["train"][execution] = t_in
#             ts_out["train"][execution] = t_out
#         elif i < nb_train + nb_test:
#             ts_in["test"][execution] = t_in
#             ts_out["test"][execution] = t_out
#         elif i < nb_train + nb_test + nb_val:
#             ts_in["val"][execution] = t_in
#             ts_out["val"][execution] = t_out
#     assert (len(ts_in["train"]) == nb_train)
#     assert (len(ts_in["test"]) == nb_test)
#     assert (len(ts_in["val"]) == nb_val)
#     return ts_in,ts_out

# def concat_and_normalize(ts_in,ts_out,executions):
#     # concatenate all the training and testing sets to a single tensor, and normalize:
#     first = True
#     for execution in executions:
#         if first:
#             tr_in = ts_in["train"][execution]
#             tr_out = ts_out["train"][execution]
#             te_in = ts_in["test"][execution]
#             te_out = ts_out["test"][execution]
#             val_in = ts_in["val"][execution]
#             val_out = ts_out["val"][execution]
#             first = False
#         else:
#             tr_in = torch.cat((tr_in, ts_in["train"][execution]))
#             tr_out = torch.cat((tr_out, ts_out["train"][execution]))
#             te_in = torch.cat((te_in, ts_in["test"][execution]))
#             te_out = torch.cat((te_out, ts_out["test"][execution]))
#             val_in = torch.cat((val_in, ts_in["val"][execution]))
#             val_out = torch.cat((val_out, ts_out["val"][execution]))
#
#     maxs = torch.cat((tr_in, te_in, val_in)).abs().max(dim=0).values
#     # maxs_te = te_in.abs().max(dim = 0).values
#
#     tr_in = torch.nan_to_num(tr_in / maxs)
#     te_in = torch.nan_to_num(te_in / maxs)
#     val_in = torch.nan_to_num(val_in / maxs)
#
#     d_ft_in = {"train": tr_in,"val": val_in,"test": te_in}
#     d_ft_out = {"train": tr_out,"val": val_out,"test": te_out}
#
#     return d_ft_in,d_ft_out

def concat_and_normalize_ext_out(ts_in, ts_out, ts_inter, executions,normalize=True):
    # concatenate all the training and testing sets to a single tensor, and normalize:
    first = True
    for execution in executions:
        if first:
            tr_in = ts_in["train"][execution]
            tr_out = ts_out["train"][execution]
            tr_inter = ts_inter["train"][execution]

            te_in = ts_in["test"][execution]
            te_out = ts_out["test"][execution]
            te_inter = ts_inter["test"][execution]

            val_in = ts_in["val"][execution]
            val_out = ts_out["val"][execution]
            val_inter = ts_inter["val"][execution]
            first = False
        else:
            tr_in = torch.cat((tr_in, ts_in["train"][execution]))
            tr_out = torch.cat((tr_out, ts_out["train"][execution]))
            tr_inter = torch.cat((tr_inter, ts_inter["train"][execution]))

            te_in = torch.cat((te_in, ts_in["test"][execution]))
            te_out = torch.cat((te_out, ts_out["test"][execution]))
            te_inter = torch.cat((te_inter, ts_inter["test"][execution]))

            val_in = torch.cat((val_in, ts_in["val"][execution]))
            val_out = torch.cat((val_out, ts_out["val"][execution]))
            val_inter = torch.cat((val_inter, ts_inter["val"][execution]))

    maxs = dict()
    maxs["in"] = torch.cat((tr_in, te_in, val_in)).abs().max(dim=0).values
    maxs["inter"] = torch.cat((tr_inter, te_inter, val_inter)).abs().max(dim=0).values
    # maxs_te = te_in.abs().max(dim = 0).values

    if normalize:
        tr_in = torch.nan_to_num(tr_in / maxs["in"])
        te_in = torch.nan_to_num(te_in / maxs["in"])
        val_in = torch.nan_to_num(val_in / maxs["in"])

        tr_inter = torch.nan_to_num(tr_inter / maxs["inter"])
        te_inter = torch.nan_to_num(te_inter / maxs["inter"])
        val_inter = torch.nan_to_num(val_inter / maxs["inter"])

    d_ft_in = {"train": tr_in, "val": val_in, "test": te_in}
    d_ft_out = {"train": tr_out, "val": val_out, "test": te_out}
    d_ft_inter = {"train": tr_inter, "val": val_inter, "test": te_inter}

    return d_ft_in, d_ft_out, d_ft_inter,maxs

def concat_and_normalize_split_by_exec(ts_in,ts_out,executions):
    # concatenate all the training and testing sets to a single tensor, and normalize:
    for set in ["train","test","val"]:
        first = True
        for execution in executions:
            if first:
                if set == "train":
                    tr_in = ts_in[set][execution]
                    tr_out = ts_out[set][execution]
                elif set == "test":
                    te_in = ts_in["test"][execution]
                    te_out = ts_out["test"][execution]
                elif set == "val":
                    val_in = ts_in["val"][execution]
                    val_out = ts_out["val"][execution]
                first = False
            else:
                if set == "train":
                    tr_in = torch.cat((tr_in, ts_in["train"][execution]))
                    tr_out = torch.cat((tr_out, ts_out["train"][execution]))
                elif set == "test":
                    te_in = torch.cat((te_in, ts_in["test"][execution]))
                    te_out = torch.cat((te_out, ts_out["test"][execution]))
                elif set == "val":
                    val_in = torch.cat((val_in, ts_in["val"][execution]))
                    val_out = torch.cat((val_out, ts_out["val"][execution]))

    maxs = torch.cat((tr_in, te_in, val_in)).abs().max(dim=0).values
    # maxs_te = te_in.abs().max(dim = 0).values
    tr_in = torch.nan_to_num(tr_in / maxs)
    te_in = torch.nan_to_num(te_in / maxs)
    val_in = torch.nan_to_num(val_in / maxs)

    d_ft_in = {"train": tr_in,"val": val_in,"test": te_in}
    d_ft_out = {"train": tr_out,"val": val_out,"test": te_out}

    return d_ft_in,d_ft_out,maxs


###############################################
#Methods for random selection of training data#
###############################################

def get_random_week_per_month_indices(df, hours_in_day=24, days_in_week=7):
    """
    Generate a list of indices, one from each month, representing one random week.

    Parameters:
        df. A dataframe, with an index assumed to be a string representing a date in format: %m-%d %H:%M:%S%z

    Returns:
        list of int: A list of indices, representing one random week for each month.
        The week always starts on the first hour of a random day.
    """
    h_in_w = hours_in_day * days_in_week

    # First, convert the index of the input df to a proper DateTime index
    dt_index = pd.to_datetime(df.index, format='%m-%d %H:%M:%S%z', utc=True).tz_localize(None) + pd.DateOffset(hours=1)

    # Group the date indices by month
    grouped = df.groupby(dt_index.to_period('M'))

    # Initialize a list to store the selected indices
    selected_indices = []
    hours_before_month_started = 0
    # Select one random week index from each month
    for group in grouped:
        assert (len(group) >= h_in_w)
        random_week_index = random.randint(0, len(group) // h_in_w - 1)

        for ix in range(random_week_index * h_in_w, random_week_index * (h_in_w) + h_in_w):
            selected_indices.append(hours_before_month_started + ix)
        hours_before_month_started += len(group)

    return np.array(selected_indices)

def get_random_days_indices(hours_available,nb_selected,hours_in_day=24,sorted = True):
    """
        Generates a list of random, non-overlapping time indices for a specified number of days.

        Parameters:
            - hours_available (int): Total number of hours available for selection.
            - nb_selected (int): Number of days to select time indices for.
            - hours_in_day (int, optional): Number of hours in a day. Default is 24.
            - sorted (bool, optional): If True, the generated indices are sorted in ascending order.
                                      If False, they are not sorted. Default is True.

        Returns:
            - np.ndarray: A numpy array containing the selected time indices.

        Raises:
            - AssertionError: Raised if hours_available is less than the product of nb_selected and hours_in_day.
    """
    assert hours_available>=nb_selected*hours_in_day

    index_list = []
    for _ in range(nb_selected):
        r = random.randint(0, hours_available)
        i = hours_in_day * round(r/hours_in_day)

        while i in index_list:
            r = random.randint(0, hours_available)
            i = hours_in_day * round(r/hours_in_day)
        this_index_list = [i+j for j in range(hours_in_day)]
        index_list.append(this_index_list)
    if sorted:
        return np.sort((np.array(index_list)).flatten())
    else:
        return np.array(index_list).flatten()

# def get_random_adj_period_slicer(nb_available,nb_selected,period_length,sorted = True):
#     start_idxs = get_random_hours_slicer(nb_available-period_length,nb_selected,sorted=sorted)
#     if sorted:
#         start_idxs = np.sort(start_idxs)
#     index_list = [[si+i for i in period_length] for si in start_idxs]
#     return index_list.flatten()

def get_random_hours_indices(nb_available,nb_selected,min_offset = 1,sorted = True):
    """
    Generates a list of random, non-overlapping indices for a specified number of hours.

    Parameters:
        - nb_available (int): Total number of available indices for selection.
        - nb_selected (int): Number of indices to select.
        - min_offset (int, optional): Minimum separation between selected indices. Default is 1.
        - sorted (bool, optional): If True, the generated indices are sorted in ascending order.
                                  If False, they are not sorted. Default is True.

    Returns:
        - np.ndarray: A numpy array containing the selected indices.

    Raises:
        - AssertionError: Raised if nb_available is less than the product of nb_selected and min_offset.
        - AssertionError: Raised if it is not possible to find non-overlapping indices within a reasonable number of attempts.
    """
    print(nb_available,nb_selected,min_offset)
    assert nb_available>=nb_selected*min_offset
    idx_l_size = 0
    index_list = []
    counter = 0
    while idx_l_size < nb_selected:
        r = random.randint(0, nb_available-1)
        after = list(range(r,r+min_offset))
        before = list(range(r-min_offset+1,r+1))
        if set(index_list).isdisjoint(before) and set(index_list).isdisjoint(after):
            idx_l_size += 1
            index_list.append(r)
            counter = 0
        counter += 1
        assert(counter <1e7)
    if sorted:
        return np.sort(index_list)
    else:
        return index_list

def return_selection(dfs_dict_list, indices):
    """
    Select specific rows from a list of dictionaries containing DataFrames.

    Given a list of dictionaries where each dictionary represents different executions with DataFrames,
    this function selects DataFrames from each execution based on the provided indices.

    Parameters:
    - dfs_dict_list (list of dict): A list of dictionaries, where each dictionary contains DataFrames.
    - indices (list or slice): Indices to select from each DataFrame in the dictionaries.

    Returns:
    - selections (list of dict): A list of dictionaries, where each dictionary contains the selected DataFrames
      based on the provided indices.

    Raises:
    - AssertionError: If the keys (executions) in the dictionaries are not equal across all dictionaries.
    """
    selections = list()
    first_dict_keys = set(dfs_dict_list[0].keys())

    for i, d in enumerate(dfs_dict_list):
        d_sel = dict()

        if set(d.keys()) != first_dict_keys:
            raise AssertionError("Keys in the dictionaries are not equal")

        for exe in first_dict_keys:
            d_sel[exe] = dfs_dict_list[i][exe].iloc[indices, :]

        selections.append(d_sel)

    return selections


#################################################
#Methods for single execution (operational cost)#
#################################################

def load_training_data(folder, period, sc, il_os=None,output = "SystemCosts",execution = "Network_Existing_Generation_Full"):
    df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0], index_col=0)
    df_out_e = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}_{output}.csv", header=[0], index_col=0)

    # Make sure all the values are floats (And Ordered Alphabetically?)
    print(len(df_in_e.columns))
    for col in df_out_e.columns:
        df_out_e[col] = df_out_e[col].astype(float)
    for col in df_in_e.columns:
        df_in_e[col] = df_in_e[col].astype(float)

    if il_os != None:
        dfs_ilo = dict()
        for il_o in il_os:
            df_inter = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}_{il_o}.csv", header=[0],
                                   index_col=0)
            for col in df_inter.columns:
                df_inter[col] = df_inter[col].astype(float)
            dfs_ilo[il_o] = df_inter
    return df_in_e,df_out_e,df_inter

def join_inter(dfs_inter):
    df_inter_j= pd.concat([dfs_inter[t] for t in sorted(dfs_inter.keys())], axis=1)

    return df_inter_j


