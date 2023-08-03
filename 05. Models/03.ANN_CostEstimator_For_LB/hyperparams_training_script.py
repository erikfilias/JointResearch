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
    folder = "../Data/RTS24_AC_12w"
    all_executions = DataLoading.list_executions(folder="../Data/RTS24_AC_12w", per=period, sc=sc)
    executions_start = 0
    executions_end = 40
    executions = all_executions[executions_start:executions_end]
    te_s = 0.3
    val_s = 0.4

    exec_name = f"All_Exec_split_by_exec_t{te_s}_v{val_s}"
    folder_to_save = f"RTS24_AC_12w_split_by_exec_{executions_start}_{executions_end}"

    #Load inputs and outputs in dataframes
    dfs_in, dfs_out = DataLoading.load_data(folder, executions, period, sc)
    #Convert to pytorch tensors
    ts_in, ts_out = DataLoading.split_tr_val_te_by_exec(dfs_in, dfs_out, executions, te_s, val_s, False)
    #Create dataloaders and store the maxs for normalisation
    d_ft_in, d_ft_out, maxs = DataLoading.concat_and_normalize_split_by_exec(ts_in, ts_out, executions)

    #Create TensorDatasets
    train = TensorDataset(d_ft_in['train'].float(), d_ft_out['train'].float())
    validation = TensorDataset(d_ft_in['val'].float(), d_ft_out['val'].float())

    #Perform the actual loop that checks multiple hyperparams
    i = 0
    nbs_hidden = [2,3]
    dors = [0,0.05]  # ,0.05,0.1]#,0.05]
    relu_outs = [False]

    batch_sizes = [32, 64, 128,256]
    learning_rates = [0.0025 * 4 ** i for i in range(1, 4, 1)]
    nbs_e = [4, 8, 16, 32, 64]  # ,8]
    negative_penalisations = [0]

    results = pd.DataFrame()

    hp_sets = ((nb_h, dor, relu_out, bs, lr, nb_e, np) for nb_h in nbs_hidden for dor in dors for relu_out in relu_outs
               for bs in batch_sizes for lr in learning_rates for nb_e in nbs_e for np in negative_penalisations)

    print("Number of hyperparameter combinations to be considered:", len(list(hp_sets)),f" {len(nbs_e)} in the epochs dim")

    for hp_set in hp_sets:
        print("Current set of hyperparameters:")
        print(hp_set)
        print("Training starts")
        nb_hidden, dor, relu_out, bs, lr, nb_e, np = hp_set[0], hp_set[1], hp_set[2], hp_set[3], hp_set[4], hp_set[5], \
                                                     hp_set[6]

        # Create training and validation loaders based on batch size
        training_loader = DataLoader(train, batch_size=bs)
        validation_loader = DataLoader(train, batch_size=bs)

        # Initialize loss functions
        loss_fn = NN_classes.create_loss_fn(penalize_negative=np)
        loss_t_mse = torch.nn.MSELoss()

        # Create model based on hyperparameter set
        m = NN_classes.create_model(nb_hidden, d_ft_in['train'].shape[1], dropout_ratio=dor, relu_out=relu_out)
        # Create model name for saving and loading
        m_name = f"OE_{nb_hidden}h_{nb_e}e_{lr}lr_{dor}dor_{np}np_{relu_out}_ro_{bs}bs"
        # Create optimizer based on learning rate
        optimizer = torch.optim.Adam(m.parameters(), lr=lr)
        # Train the actual model
        t_start_train = time.perf_counter()
        train_loss_1 = \
        training_methods.train_multiple_epochs(nb_e, m, training_loader, validation_loader, loss_fn, optimizer, m_name,
                                               folder_to_save)[0]
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
            test_loss = loss_fn(test_predictions.squeeze(), d_ft_out["test"])
            test_loss_t_mse = loss_t_mse(test_predictions.squeeze(), d_ft_out["test"])

            train_predictions = m(d_ft_in["train"].float())
            train_loss = loss_fn(train_predictions.squeeze(), d_ft_out["train"])
            train_loss_t_mse = loss_t_mse(train_predictions.squeeze(), d_ft_out["train"])

            validation_prediction = m(d_ft_in["val"].float())
            validation_loss = loss_fn(validation_prediction.squeeze(), d_ft_out["val"])
            validation_loss_t_mse = loss_t_mse(validation_prediction.squeeze(), d_ft_out["val"])
            t_stop_eval = time.perf_counter()

            # Calculate some calculation times
            t_train = t_stop_train - t_start_train
            t_eval = t_stop_eval - t_start_eval

            # Finally, save all desired values in a dataframe
            r = pd.DataFrame({"Model_type": nb_hidden,
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
                              "Tr_l_ret": train_loss_1.item(),
                              "Train_time": t_train,
                              "Eval_time": t_eval,
                              "Test size": te_s,
                              "Val size": val_s
                              }
                             , index=[i])
            i += 1
            results = pd.concat([results, r])
        results.to_csv(f"Loss_results_csv/{exec_name}.csv")

    results.to_csv(f"Loss_results_csv/{exec_name}.csv")
