# Define the neural network model
import torch


class ObjectiveEstimator_ANN_Single_layer(torch.nn.Module):
    def __init__(self, input_size,hidden_sizes, output_size,dropout_ratio=0.0):
        super().__init__()
        self.output_layer = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_ratio)

        # define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        input = self.dropout(input)
        output = self.output_layer(input)
        return output

class ObjectiveEstimator_ANN_1hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size1, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        output = self.output_layer(hidden1_dropout)
        return output

class ObjectiveEstimator_ANN_2hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size2, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        output = self.output_layer(hidden2_dropout)
        return output

class ObjectiveEstimator_ANN_3hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        hidden_size3 = hidden_sizes[2]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size3, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2_dropout))
        hidden3_dropout = self.dropout(hidden3)
        output = self.output_layer(hidden3_dropout)
        return output

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
