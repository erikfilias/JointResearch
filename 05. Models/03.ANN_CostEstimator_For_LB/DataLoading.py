import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np

def list_executions(per,sc,folder):
    filenames = os.listdir(f"{folder}")
    executions = [fn.split(per)[0].split(sc)[1][1:-1] for fn in filenames]
    return np.unique(executions)

def load_data(folder,executions,period,sc):
    dfs_in = dict()
    dfs_out = dict()
    for execution in executions:
        # Read the data from desired execution
        df_in_e = pd.read_csv(f"{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0],index_col=0)
        df_out_e = pd.read_csv(f"{folder}/output_f_{sc}_{execution}_{period}.csv", header=[0],index_col=0)

        print(f"input_f_{sc}_{execution}_{period}.csv")

        # And order the variables:

        print(len(df_in_e.columns))
        for col in df_out_e.columns:
            df_out_e[col] = df_out_e[col].astype(float)
        for col in df_in_e.columns:
            df_in_e[col] = df_in_e[col].astype(float)
        dfs_in[execution] = df_in_e
        dfs_out[execution] = df_out_e
    return dfs_in,dfs_out

def split_tr_val_te(dfs_in,dfs_out,executions,te_s,val_s):
    ts_in = dict()
    ts_out = dict()

    ts_in["train"] = dict()
    ts_in["test"] = dict()
    ts_in["val"] = dict()

    ts_out["train"] = dict()
    ts_out["test"] = dict()
    ts_out["val"] = dict()

    # Test size as fraction of full dataset, validation size as fraction of training data set
    test_size, validation_size = te_s, val_s

    for execution in executions:
        # Convert input dataframes numpy arrays sum the columns of the output:
        np_in = dfs_in[execution].to_numpy()
        np_out = dfs_out[execution].to_numpy().sum(axis=1)

        # We don't normalize the separate runs, but will do it afterward, all together

        # Convert to torch tensors
        t_in = torch.from_numpy(np_in)
        t_out = torch.from_numpy(np_out)

        # And split into train, validation, and test set:
        train_in, ts_in["test"][execution], train_out, ts_out["test"][execution] = train_test_split(t_in, t_out,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=False)
        ts_in["train"][execution], ts_in["val"][execution], ts_out["train"][execution], ts_out["val"][
            execution] = train_test_split(train_in, train_out, test_size=validation_size, shuffle=False)
    return ts_in,ts_out

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

    return d_ft_in,d_ft_out