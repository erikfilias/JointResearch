# Define the neural network model
import torch
import math


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
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0,relu_out = False ):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size1, output_size)
        self.relu_out = relu_out

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden1_dropout))
        else:
            output = self.output_layer(hidden1_dropout)
        return output

class ObjectiveEstimator_ANN_2hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.,relu_out = False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size2, output_size)

        self.relu_out = relu_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden2_dropout))
        else:
            output = self.output_layer(hidden2_dropout)
        return output

class ObjectiveEstimator_ANN_3hidden_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0,relu_out =False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        hidden_size3 = hidden_sizes[2]
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size3, output_size)
        self.relu_out = relu_out

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2_dropout))
        hidden3_dropout = self.dropout(hidden3)

        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden3_dropout))
        else:
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

def create_model(nb_hidden, input_size, dropout_ratio,relu_out=False):
    hidden_sizes = []
    if nb_hidden == 0:
        hidden_sizes.append(input_size)
    elif nb_hidden == 1:
        hidden_sizes.extend([int(math.sqrt(input_size))])
    elif nb_hidden == 2:
        hidden_sizes.extend([int(math.sqrt(input_size)), int(math.sqrt(math.sqrt(input_size)))])
    elif nb_hidden == 3:
        hidden_sizes.extend([int(input_size / 4), int(input_size / 16), int(input_size / 64)])



    if nb_hidden == 0:
        model_class = ObjectiveEstimator_ANN_Single_layer
    elif nb_hidden == 1:
        model_class = ObjectiveEstimator_ANN_1hidden_layer
    elif nb_hidden == 2:
        model_class = ObjectiveEstimator_ANN_2hidden_layer
    elif nb_hidden == 3:
        model_class = ObjectiveEstimator_ANN_3hidden_layer
    model = model_class(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1, dropout_ratio=dropout_ratio,relu_out=relu_out)
    print(model,dropout_ratio,nb_hidden,relu_out)
    return model

def create_loss_fn(penalize_negative=0):
    def my_loss(output, target):
        MSE_l = torch.mean((output - target) ** 2)
        negative_penalization = torch.nan_to_num(torch.mean(-1 * penalize_negative * (output[output < 0])))
        # if negative_penalization != 0:
        #     print(f"********************")
        #     print(f"********************")
        #
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #
        #     print(f"NP = {negative_penalization}, MSE_l ={MSE_l} ")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        #     print(f"********************")
        if negative_penalization != 0:
            print(f"NP = {negative_penalization}, MSE_l ={MSE_l} ")

        loss = MSE_l + negative_penalization
        return loss

    return my_loss
