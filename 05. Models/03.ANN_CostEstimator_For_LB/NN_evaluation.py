import pandas as pd
import torch
import numpy as np
import NN_classes

def calculate_lb_from_dfs_out(dfs_out, execution,as_numpy = True):
    if as_numpy:
        return (dfs_out["Network_Existing_Generation_Full"].sum(axis=1) - dfs_out[execution].sum(axis=1)).to_numpy()
    else:
        return (dfs_out["Network_Existing_Generation_Full"].sum(axis=1) - dfs_out[execution].sum(axis=1))


def calculate_lb_from_ts_out(ts_out, ex):
    b = "Network_Existing_Generation_Full"
    all_ts_out_ex = torch.concat((ts_out["train"][ex], ts_out["test"][ex], ts_out["val"][ex]))
    all_ts_out_benchmark = torch.concat((ts_out["train"][b], ts_out["test"][b], ts_out["val"][b]))
    return all_ts_out_benchmark - all_ts_out_ex


def find_xthbest_model_params_from_df(df_losses, loss_to_sort, xth_best=1):
    return df_losses.sort_values(by=loss_to_sort)[xth_best - 1:xth_best]


def extract_model_params_from_row(row):
    model_type = row.Model_type.item()
    model_type = tuple(map(int, model_type.replace("(", "").replace(")", "").split(', ')))
    dor = row.Dor.item()
    lr = row.Lr.item()
    nb_e = row.Epochs.item()

    relu_out = row.Relu_out.item()
    np = row.Np.item()
    bs = row.Batch_size.item()
    alpha = row.alpha.item()
    MAE = row.MAE.item()

    return {"Model_type": model_type, "nb_e": nb_e, "lr": lr, "dor": dor, "np": np, "ro": relu_out, "bs": bs,
            "alpha": alpha, "MAE": MAE}


def create_model_and_load_state_from_row(row, input_size, inter_size, hyperloop_name, cluster_run=True):
    # First, extract params from row

    model_type = row.Model_type.item()
    model_type = tuple(map(int, model_type.replace("(", "").replace(")", "").split(', ')))
    dor = row.Dor.item()
    lr = row.Lr.item()
    nb_e = row.Epochs.item()

    relu_out = row.Relu_out.item()
    np = row.Np.item()
    bs = row.Batch_size.item()
    alpha = row.alpha.item()
    MAE = row.MAE.item()

    if str(alpha) == "0.0":
        alpha = "0"
    if str(dor) == "0.0":
        dor = "0"

    if row.Min_val.item():
        mt = "min_val"
    else:
        mt = "all_epochs"

    # Then create model of given type
    m = NN_classes.create_model(model_type, input_size, dropout_ratio=dor, relu_out=relu_out, inter=True,
                                inter_size=inter_size)

    # Finally, extract model state from dict

    # m_name = f"OE_{model_type}h_{nb_e}e_{lr}lr_{dor}dor_{np}np_{relu_out}_ro_{bs}bs"
    m_name = f"OE_{model_type}h_{nb_e}e_{lr}lr_{dor}dor_{np}np_{relu_out}ro_{bs}bs_{alpha}ill_{MAE}MAE"

    if cluster_run:
        # m_name = f"OE_{model_type}h_{nb_e}e_{lr}lr_{dor}dor_{np}np_{relu_out}_ro_{bs}bs"
        path = f"ResultsClusterRuns/trained_models/{hyperloop_name}/{mt}/model_{m_name}.pth"
    else:
        path = f"trained_models/{hyperloop_name}/{mt}/model_{m_name}.pth"

    m.load_state_dict(torch.load(path))
    m.eval()

    return m


def get_lb_est_and_actual(m, ex, dfs_in, dfs_out):
    negf = all_executions[0]
    ex_in_e = torch.nan_to_num(dfs_in[ex].to_numpy() / maxs["in"])
    ex_in_negf = torch.nan_to_num(dfs_in[negf].to_numpy() / maxs["in"])

    prediction_e = m(ex_in_e.float())[0].detach().numpy()
    prediction_negf = m(ex_in_negf.float())[0].detach().numpy()

    lb_est = prediction_negf - prediction_e
    lb_actual = calculate_lb_from_dfs_out(dfs_out, ex).to_numpy()
    return lb_est.flatten(), lb_actual.flatten()


def get_NN_estimates_from_dfs_in(m, ex, dfs_in):
    ex_in_e = torch.nan_to_num(dfs_in[ex].to_numpy() / maxs["in"])
    prediction_e = m(ex_in_e.float())[0].detach().numpy()
    return prediction_e.flatten()


def get_actual_from_dfs_out(ex, dfs_out,as_numpy =True):
    if as_numpy:
        actual = dfs_out[ex].sum(axis=1).to_numpy()
    else:
        actual = dfs_out[ex].sum(axis=1)
    return actual

