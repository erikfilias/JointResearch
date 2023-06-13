# Define the neural network model
import torch

class ObjectiveEstimator_ANN_Single_layer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_layer = torch.nn.Linear(input_size, output_size)
        # define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        output = self.output_layer(input)
        return output
class ObjectiveEstimator_ANN_3hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        torch.nn.init.kaiming_uniform_(self.hidden_layer1.weight, a=0)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.output_layer = torch.nn.Linear(hidden_size3, output_size)

        # define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden2 = torch.relu(self.hidden_layer2(hidden1))
        # hidden3 = torch.relu(self.hidden_layer3(hidden2))
        hidden3 = self.hidden_layer3(hidden2)
        output = self.output_layer(hidden3)
        return output
class ObjectiveEstimator_ANN_2hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        torch.nn.init.kaiming_uniform_(self.hidden_layer1.weight, a=0)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = torch.nn.Linear(hidden_size2, output_size)

        # define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden2 = torch.relu(self.hidden_layer2(hidden1))
        # hidden3 = torch.relu(self.hidden_layer3(hidden2))
        output = self.output_layer(hidden2)
        return output
class ObjectiveEstimator_ANN_1hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.output_layer = torch.nn.Linear(hidden_size1, output_size)

        # define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        # hidden3 = torch.relu(self.hidden_layer3(hidden2))
        output = self.output_layer(hidden1)
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