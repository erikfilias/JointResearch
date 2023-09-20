import DataLoading
from torch.utils.data import DataLoader,TensorDataset
import torch
import pandas as pd
import NN_classes
import training_methods
import time

if __name__ == '__main__':
    #Some initialising stuff
    sc = "sc01"
    period = "2030"
    #folder = "../Data/RTS24_AC_12w_ext_o_dummy"
    # folder = "../Data/3-bus_AC_12w_ext_o_dummy_LCOE"
    folder = "../Data/9n_AC_12w_ext_o_dummy_LCOE"

    all_executions = DataLoading.list_executions(folder=folder, per=period, sc=sc)
    executions_start = 0
    executions_end = len(all_executions)
    executions = all_executions[executions_start:executions_end]
    te_s = 0.3
    val_s = 0.4
    outp = "SystemCosts"

    #exec_name = f"RTS24_AC_12w_dummy_{te_s}_v{val_s}_PF_LCOE_{executions_start}_{executions_end}"
    #exec_name = f"3-bus_AC_12w_dummy_{te_s}_v{val_s}_PF_LCOE_{executions_start}_{executions_end}"
    exec_name = f"9n_AC_12w_dummy_{te_s}_v{val_s}_PF_LCOE_{executions_start}_{executions_end}_v2"

    folder_to_save = f"{exec_name}"

    #Load inputs and outputs in dataframes
    dfs_in, dfs_out, dfs_inter = DataLoading.load_data_ext_out(folder, executions, period, sc, ["PowerFlow"],outp)
    dfs_inter_j = DataLoading.join_frames_inter_layer(dfs_inter)
    dfs_inter_j = DataLoading.trim_columns_to_common(dfs_inter_j)

    # Convert to pytorch tensors
    ts_in, ts_out, ts_inter = DataLoading.split_tr_val_te_ext_out(dfs_in, dfs_out, dfs_inter_j, executions, te_s, val_s)

    #Create dataloaders
    d_ft_in, d_ft_out, d_ft_inter,maxs = DataLoading.concat_and_normalize_ext_out(ts_in, ts_out, ts_inter, executions)

    #Create TensorDatasets
    train = TensorDataset(d_ft_in['train'].float(), d_ft_out['train'].float(), d_ft_inter['train'])
    validation = TensorDataset(d_ft_in['val'].float(), d_ft_out['val'].float(), d_ft_inter['val'].float())

    #Perform the actual loop that checks multiple hyperparams
    i = 0
    nbs_hidden = [(2, 0),(3, 0),(3, 1),(3, 2)] #
    #nbs_hidden = [(2, 0),(3,2)]
    #dors = [0,0.05,0.25]  # ,0.05,0.1]#,0.05]
    dors = [0,0.05,0.1,0.2]
    #dors = [0]
    #relu_outs = [False,True]
    relu_outs = [True]

    batch_sizes = [64]
    #learning_rates = [0.0025 * 4 ** i for i in range(-2, 0, 1)]
    learning_rates = [0.0025 * 4 ** i for i in range(-1, 2, 1)]
    #nbs_e = [32,64,128]  # ,8]
    nbs_e = [8,16,32,64,128]  # ,8]
    negative_penalisations = [0]
    #alphas = [0, 0.01,0.04,0.16]
    alphas = [0,0.04,0.16]
    #alphas = [0]

    beta = 1
    MAEs = [True,False]
    #MAEs = [True]
    results = pd.DataFrame()

    hp_sets = ((nb_h, dor, relu_out, bs, lr, nb_e, np, alpha,MAE) for nb_h in nbs_hidden for dor in dors for relu_out in
               relu_outs for bs in batch_sizes for lr in learning_rates for nb_e in nbs_e for np in
               negative_penalisations for alpha in alphas for MAE in MAEs)

    inter_size = dfs_inter_j["Network_Existing_Generation_Full"].shape[1]

    print(f"Number of hyperparameter combinations to be considered: {len(list(hp_sets))}, {len(nbs_e)} in the epochs dim")
    hp_sets = ((nb_h, dor, relu_out, bs, lr, nb_e, np, alpha,MAE) for nb_h in nbs_hidden for dor in dors for relu_out in
               relu_outs for bs in batch_sizes for lr in learning_rates for nb_e in nbs_e for np in
               negative_penalisations for alpha in alphas for MAE in MAEs)
    print("test")
    counter = 0
    for hp_set in hp_sets:
        counter+=1
        print(hp_set, "Counter:",counter)
        nb_hidden, dor, relu_out, bs, lr, nb_e, np, alpha,MAE = hp_set[0], hp_set[1], hp_set[2], hp_set[3], hp_set[4], hp_set[
            5], hp_set[6], hp_set[7],hp_set[8]

        # Create training and validation loaders based on batch size
        training_loader = DataLoader(train, batch_size=bs)
        validation_loader = DataLoader(validation, batch_size=bs)

        # Initialize loss functions
        loss_fn = NN_classes.create_custom_loss(alpha=alpha, beta=beta,MAE=MAE)
        loss_t_mse = torch.nn.MSELoss()
        loss_mae = torch.nn.L1Loss()

        # Create model based on hyperparameter set
        m = NN_classes.create_model(nb_hidden, d_ft_in['train'].shape[1], dropout_ratio=dor, relu_out=relu_out, inter=True,
                                    inter_size=inter_size)
        # Create model name for saving and loading
        m_name = f"OE_{nb_hidden}h_{nb_e}e_{lr}lr_{dor}dor_{np}np_{relu_out}ro_{bs}bs_{alpha}ill_{MAE}MAE"
        # Create optimizer based on learning rate
        optimizer = torch.optim.Adam(m.parameters(), lr=lr)
        # Train the actual model
        t_start_train = time.perf_counter()
        train_loss_1 = \
        training_methods.train_multiple_epochs(nb_e, m, training_loader, validation_loader, loss_fn, optimizer, m_name,
                                               folder_to_save, True)[0]
        t_stop_train = time.perf_counter()

        # In the following loop, we retreive the models from saved locations and calculate losses
        for mt in ["min_val", "all_epochs"]:
            t_start_eval = time.perf_counter()
            path = f"trained_models/{folder_to_save}/{mt}/model_{m_name}.pth"

            # Retreive model state and set to evaluation mode
            m.load_state_dict(torch.load(path))
            m.eval()

            # Calculate losses
            test_predictions = m(d_ft_in["test"].float())
            test_loss = loss_fn(test_predictions[0].squeeze(), d_ft_out["test"], test_predictions[1].squeeze(),
                                d_ft_inter["test"])
            test_loss_t_mse = loss_t_mse(test_predictions[0].squeeze(), d_ft_out["test"])
            test_loss_mae = loss_mae(test_predictions[0].squeeze(), d_ft_out["test"])

            train_predictions = m(d_ft_in["train"].float())
            train_loss = loss_fn(train_predictions[0].squeeze(), d_ft_out["train"], train_predictions[1].squeeze(),
                                 d_ft_inter["train"])
            train_loss_t_mse = loss_t_mse(train_predictions[0].squeeze(), d_ft_out["train"])
            train_loss_mae = loss_mae(train_predictions[0].squeeze(), d_ft_out["train"])

            validation_prediction = m(d_ft_in["val"].float())
            validation_loss = loss_fn(validation_prediction[0].squeeze(), d_ft_out["val"],
                                      validation_prediction[1].squeeze(), d_ft_inter["val"])
            validation_loss_t_mse = loss_t_mse(validation_prediction[0].squeeze(), d_ft_out["val"])
            validation_loss_mae = loss_mae(validation_prediction[0].squeeze(), d_ft_out["val"])

            t_stop_eval = time.perf_counter()

            # Calculate some calculation times
            t_train = t_stop_train - t_start_train
            t_eval = t_stop_eval - t_start_eval


            # Finally, save all desired values in a dataframe
            r = pd.DataFrame({"Model_type": [nb_hidden],
                              "Dor": dor,
                              "Relu_out": relu_out,
                              "Batch_size": bs,
                              "Lr": lr,
                              "Epochs": nb_e,
                              "Np": np,
                              "Min_val": mt,
                              "Tr_l": train_loss.item(),
                              "Te_l": test_loss.item(),
                              "V_l": validation_loss.item(),
                              "Tr_l_t_mse": train_loss_t_mse.item(),
                              "Te_l_t_mse": test_loss_t_mse.item(),
                              "V_l_t_mse": validation_loss_t_mse.item(),
                              "Tr_l_mae": train_loss_mae.item(),
                              "Te_l_mae": test_loss_mae.item(),
                              "V_l_mae": validation_loss_mae.item(),
                              "Tr_l_ret": train_loss_1.item(),
                              "Train_time": t_train,
                              "Eval_time": t_eval,
                              "alpha": alpha,
                              "beta": beta,
                              "MAE": MAE,
                              "Test size": te_s,
                              "Val size": val_s
                              }
                             , index=[i])
            i += 1
            results = pd.concat([results, r])
        results.to_csv(f"Loss_results_csv/{exec_name}.csv")

    results.to_csv(f"Loss_results_csv/{exec_name}.csv")
