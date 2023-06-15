import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data(folder,executions,period,sc):
    dfs_in = dict()
    dfs_out = dict()
    for execution in executions:
        # Read the data from desired execution
        df_in_e = pd.read_csv(f"Data/{folder}/input_f_{sc}_{execution}_{period}.csv", header=[0, 1])
        df_out_e = pd.read_csv(f"Data/{folder}/output_f_{sc}_{execution}_{period}.csv", header=[0, 1])

        # Drop the first row(s) because its useless
        df_in_e = df_in_e.drop([0])
        if (folder == "RTS_24") or (folder == ""):
            df_out_e = df_out_e.drop([0])
        elif (folder == "RTS_24_AC") :
            df_out_e = df_out_e.drop([0, 1])
        print(f"Data/input_f_{sc}_{execution}_{period}.csv")

        # Split the real and imaginary part the input data:
        df_in_e_r = df_in_e["Value_R"]
        df_in_e_i = df_in_e["Value_I"]

        # And order the variables:

        print(len(df_in_e_r.columns) + len(df_in_e_i.columns))

        df_in_e_c = pd.concat([df_in_e_r, df_in_e_i], axis=1)
        df_out_e = df_out_e["Value"]
        for col in df_out_e.columns:
            df_out_e[col] = df_out_e[col].astype(float)
        dfs_in[execution] = df_in_e_c
        dfs_out[execution] = df_out_e
    return dfs_in,dfs_out

def split_tr_val_te(dfs_in,dfs_out,executions):
    ts_in = dict()
    ts_out = dict()

    ts_in["train"] = dict()
    ts_in["test"] = dict()
    ts_in["val"] = dict()

    ts_out["train"] = dict()
    ts_out["test"] = dict()
    ts_out["val"] = dict()

    # Test size as fraction of full dataset, validation size as fraction of training data set
    test_size, validation_size = 0.2, 0.2

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