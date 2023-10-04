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

def load_data_input_only(folder, executions, period, sc):
    dfs_in = dict()

    for execution in executions:
        # Read the data from desired execution
        df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0], index_col=0)

        print(f"input_f_{sc}_{execution}_{period}.csv")

        print(len(df_in_e.columns))
        for col in df_in_e.columns:
            df_in_e[col] = df_in_e[col].astype(float)

        dfs_in[execution] = df_in_e

    return dfs_in

def join_frames_inter_layer(dfs_inter):
    dfs_inter_j = dict()
    for execution in dfs_inter.keys():
        dfs_inter_j[execution] = pd.concat([dfs_inter[execution][t] for t in dfs_inter[execution].keys()],axis=1)
    return dfs_inter_j

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

def split_tr_val_te_by_exec(dfs_in,dfs_out,executions,te_s,val_s,randomized = False):
    ts_in = dict()
    ts_out = dict()

    ts_in["train"] = dict()
    ts_in["test"] = dict()
    ts_in["val"] = dict()

    ts_out["train"] = dict()
    ts_out["test"] = dict()
    ts_out["val"] = dict()

    nb_executions = len(dfs_in)

    # Test size as fraction of full dataset, validation size as fraction of training data set
    nb_executions = len(dfs_in)
    nb_test = int(round(nb_executions * te_s))
    nb_val = int(round((nb_executions - nb_test) * val_s))
    nb_train = int(round((nb_executions - nb_test) * (1 - val_s)))
    assert (nb_executions == nb_test + nb_train + nb_val)

    l_keys = list(dfs_in)
    if randomized:
        random.shuffle(dfs_in)

    for i, execution in enumerate(executions):
        # Convert input dataframes numpy arrays sum the columns of the output:
        np_in = dfs_in[execution].to_numpy()
        np_out = dfs_out[execution].to_numpy().sum(axis=1)
        # Convert to torch tensors
        t_in = torch.from_numpy(np_in)
        t_out = torch.from_numpy(np_out)

        if i < nb_train:
            ts_in["train"][execution] = t_in
            ts_out["train"][execution] = t_out
        elif i < nb_train + nb_test:
            ts_in["test"][execution] = t_in
            ts_out["test"][execution] = t_out
        elif i < nb_train + nb_test + nb_val:
            ts_in["val"][execution] = t_in
            ts_out["val"][execution] = t_out
    assert (len(ts_in["train"]) == nb_train)
    assert (len(ts_in["test"]) == nb_test)
    assert (len(ts_in["val"]) == nb_val)
    return ts_in,ts_out

def concat_and_normalize(ts_in,ts_out,executions):
    # concatenate all the training and testing sets to a single tensor, and normalize:
    first = True
    for execution in executions:
        if first:
            tr_in = ts_in["train"][execution]
            tr_out = ts_out["train"][execution]
            te_in = ts_in["test"][execution]
            te_out = ts_out["test"][execution]
            val_in = ts_in["val"][execution]
            val_out = ts_out["val"][execution]
            first = False
        else:
            tr_in = torch.cat((tr_in, ts_in["train"][execution]))
            tr_out = torch.cat((tr_out, ts_out["train"][execution]))
            te_in = torch.cat((te_in, ts_in["test"][execution]))
            te_out = torch.cat((te_out, ts_out["test"][execution]))
            val_in = torch.cat((val_in, ts_in["val"][execution]))
            val_out = torch.cat((val_out, ts_out["val"][execution]))

    maxs = torch.cat((tr_in, te_in, val_in)).abs().max(dim=0).values
    # maxs_te = te_in.abs().max(dim = 0).values
    tr_in = torch.nan_to_num(tr_in / maxs)
    te_in = torch.nan_to_num(te_in / maxs)
    val_in = torch.nan_to_num(val_in / maxs)

    d_ft_in = {"train": tr_in,"val": val_in,"test": te_in}
    d_ft_out = {"train": tr_out,"val": val_out,"test": te_out}

    return d_ft_in,d_ft_out

def concat_and_normalize_ext_out(ts_in, ts_out, ts_inter, executions):
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
        for execution in ts_in[set].keys():
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

def trim_columns_to_common(dfs_inter_j):
    common_columns = list(set.intersection(*(set(df.columns) for df in dfs_inter_j.values())))
    # Filter DataFrames to keep only common columns
    filtered_dataframes_dict = {name: df[common_columns] for name, df in dfs_inter_j.items()}
    return filtered_dataframes_dict

###############################################
#Methods for random selection of training data#
###############################################

def get_random_week_each_month():
    pass
def get_random_days():
    pass

def get_random_adj_period_slicer(nb_available,nb_selected,period_length,sorted = True):
    start_idxs = get_random_hours_slicer(nb_available-period_length,nb_selected,sorted=sorted)
    if sorted:
        start_idxs = np.sort(start_idxs)
    index_list = [[si+i for i in period_length] for i in start_idxs]
    return index_list.flatten()






def get_random_hours_slicer(nb_available,nb_selected,sorted = True):
    index_list = [random.randint(0,nb_available) for i in range(0,nb_selected)]
    if sorted:
        return np.sort(index_list)
    else:
        return index_list



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
    df_inter_j= pd.concat([dfs_inter[t] for t in dfs_inter.keys()], axis=1)

    return df_inter_j


