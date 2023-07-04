import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train_and_get_loss(model,tr_in,tr_out,nb_epochs,lr,print_ = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        train_predictions = model(tr_in.float())
        train_loss = torch.nn.MSELoss()(train_predictions.float().squeeze(), tr_out.float())

        # Backward pass
        # optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        #Print the training loss every 10 epochs
        if print_ and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}')
    train_predictions = model(tr_in.float())
    train_loss = torch.nn.MSELoss()(train_predictions.float().squeeze(), tr_out.float())
    return train_loss


def train_one_epoch(model, training_loader, epoch_index, optimizer, loss_fn,f_print = 100):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    losses = []
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.unsqueeze(1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        losses.append(loss.item())
        if i % f_print == 0:
            last_loss = running_loss / f_print # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return losses

def train_multiple_epochs(nb_epochs,model,training_loader,validation_loader,loss_fn,optimizer,model_name, folder=None):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    epoch_number = 0

    best_vloss = 1_000_000.
    for epoch in range(nb_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        one_epoch_losses = train_one_epoch(model,training_loader,epoch_number,optimizer,loss_fn)
        avg_loss = np.mean(one_epoch_losses)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(np.mean(avg_loss), avg_vloss))

        # # Log the running loss averaged per batch
        # # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            if not(folder == None):
                min_val_model_path = 'trained_models/{}/min_val/model_{}.pth'.format(folder,model_name)
                torch.save(model.state_dict(), min_val_model_path)





        epoch_number += 1
    model_path = 'trained_models/{}/all_epochs/model_{}.pth'.format(folder,model_name)
    if not (folder == None):
        torch.save(model.state_dict(), model_path)

    return best_vloss,model_path,model