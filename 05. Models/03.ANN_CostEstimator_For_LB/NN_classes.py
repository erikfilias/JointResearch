import torch
import math
import torch.nn.functional as F

######################
##Model constructors #
######################

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


class ObjectiveEstimator_ANN_inter_0_0(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0, relu_out=False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        # print(hidden_size1)
        # print(output_size)
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
        return output, hidden1

class ObjectiveEstimator_ANN_inter_1_0(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0., relu_out=False):
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
        return output, hidden2

class ObjectiveEstimator_ANN_inter_1_1(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0., relu_out=False):
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
        return output, hidden2

class ObjectiveEstimator_ANN_inter_2_0(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0, relu_out=False):
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
        return output, hidden3

class ObjectiveEstimator_ANN_inter_3_0(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0, relu_out=False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        hidden_size3 = hidden_sizes[2]
        hidden_size4 = hidden_sizes[3]

        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.hidden_layer4 = torch.nn.Linear(hidden_size3, hidden_size4)

        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size4, output_size)
        self.relu_out = relu_out

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2_dropout))
        hidden3_dropout = self.dropout(hidden3)
        hidden4 = torch.relu(self.hidden_layer4(hidden3_dropout))
        hidden4_dropout = self.dropout(hidden4)

        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden4_dropout))
        else:
            output = self.output_layer(hidden4_dropout)
        return output, hidden4

class ObjectiveEstimator_ANN_inter_3_1(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0, relu_out=False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        hidden_size3 = hidden_sizes[2]
        hidden_size4 = hidden_sizes[3]
        hidden_size5 = hidden_sizes[4]

        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.hidden_layer4 = torch.nn.Linear(hidden_size3, hidden_size4)
        self.hidden_layer5 = torch.nn.Linear(hidden_size4, hidden_size5)

        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size5, output_size)
        self.relu_out = relu_out

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2_dropout))
        hidden3_dropout = self.dropout(hidden3)
        hidden4 = torch.relu(self.hidden_layer4(hidden3_dropout))
        hidden4_dropout = self.dropout(hidden4)
        hidden5 = torch.relu(self.hidden_layer5(hidden4_dropout))
        hidden5_dropout = self.dropout(hidden5)

        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden5_dropout))
        else:
            output = self.output_layer(hidden5_dropout)
        return output, hidden4

class ObjectiveEstimator_ANN_inter_3_2(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_ratio=0.0, relu_out=False):
        super().__init__()
        hidden_size1 = hidden_sizes[0]
        hidden_size2 = hidden_sizes[1]
        hidden_size3 = hidden_sizes[2]
        hidden_size4 = hidden_sizes[3]
        hidden_size5 = hidden_sizes[4]
        hidden_size6 = hidden_sizes[5]

        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.hidden_layer4 = torch.nn.Linear(hidden_size3, hidden_size4)
        self.hidden_layer5 = torch.nn.Linear(hidden_size4, hidden_size5)
        self.hidden_layer6 = torch.nn.Linear(hidden_size5, hidden_size6)

        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.output_layer = torch.nn.Linear(hidden_size6, output_size)
        self.relu_out = relu_out

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden1 = torch.relu(self.hidden_layer1(input))
        hidden1_dropout = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1_dropout))
        hidden2_dropout = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2_dropout))
        hidden3_dropout = self.dropout(hidden3)
        hidden4 = torch.relu(self.hidden_layer4(hidden3_dropout))
        hidden4_dropout = self.dropout(hidden4)
        hidden5 = torch.relu(self.hidden_layer5(hidden4_dropout))
        hidden5_dropout = self.dropout(hidden5)
        hidden6 = torch.relu(self.hidden_layer6(hidden5_dropout))
        hidden6_dropout = self.dropout(hidden6)

        if (self.relu_out):
            output = torch.relu(self.output_layer(hidden6_dropout))
        else:
            output = self.output_layer(hidden6_dropout)
        return output, hidden4

def create_model_OC_only(nb_hidden, input_size, dropout_ratio,relu_out):
    print("Creating model objective estimator only")
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
    #print(model,dropout_ratio,nb_hidden,relu_out)
    print("Model layers: ", model)
    return model

def create_model_inter(nb_hidden,input_size,inter_size,dropout_ratio,relu_out =False,hidden_sizes = None):
    #assert(type(hidden_sizes) == tuple)
    #print(nb_hidden)
    if hidden_sizes == None:
        hidden_sizes = []
        if nb_hidden == (0,0):
            hidden_sizes.append(inter_size)
        elif nb_hidden == (1,0):
            hidden_sizes.extend([int((input_size + inter_size)/2),inter_size])
        elif nb_hidden == (1,1):
            hidden_sizes.extend([int((input_size + inter_size)/2),inter_size,int((1 + inter_size)/2)])
        elif nb_hidden == (2,0):
            hidden_sizes.extend([int(input_size)*2, int((input_size + inter_size)/2), inter_size])
        elif nb_hidden == (3,0):
            hidden_sizes.extend([min(60,int(input_size*2)), min(60,int(input_size *3)), int((input_size + inter_size)/2),inter_size])
        elif nb_hidden == (3,1):
            hidden_sizes.extend([min(60,int(input_size*2)), min(60,int(input_size *3)), int((input_size + inter_size)/2),inter_size,max(int((inter_size)/2),1)])
        elif nb_hidden == (3,2):
            hidden_sizes.extend([min(60,int(input_size*2)), min(60,int(input_size *3)), int((input_size + inter_size)/2),inter_size,int((inter_size)/2),max(1,int((inter_size)/4))])
    print(hidden_sizes)
    if nb_hidden == (0,0):
        model_class = ObjectiveEstimator_ANN_inter_0_0
    elif nb_hidden == (1,0):
        model_class = ObjectiveEstimator_ANN_inter_1_0
    elif nb_hidden == (1,1):
        model_class = ObjectiveEstimator_ANN_inter_1_1
    elif nb_hidden == (2,0):
        model_class = ObjectiveEstimator_ANN_inter_2_0
    elif nb_hidden == (3,0):
        model_class = ObjectiveEstimator_ANN_inter_3_0
    elif nb_hidden == (3,1):
        model_class = ObjectiveEstimator_ANN_inter_3_1
    elif nb_hidden == (3,2):
        model_class = ObjectiveEstimator_ANN_inter_3_2
    model = model_class(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1, dropout_ratio=dropout_ratio,relu_out=relu_out)
    #print(model,dropout_ratio,nb_hidden,relu_out)
    return model

def create_model(nb_hidden, input_size, dropout_ratio,relu_out=False,inter =False,hidden_sizes=None,inter_size = None):
    if not(inter):
        model = create_model_OC_only(nb_hidden=nb_hidden, input_size=input_size, dropout_ratio=dropout_ratio,relu_out=relu_out)
    if inter:
        if inter_size == None:
            raise ValueError("Please provide a valid size for the intermediate layer")
        model = create_model_inter(nb_hidden=nb_hidden, input_size=input_size, dropout_ratio=dropout_ratio,hidden_sizes=hidden_sizes,inter_size=inter_size)
    return model

##################
##Loss functions #
##################
# def create_loss_fn(penalize_negative=0):
#     def my_loss(output, target):
#         MSE_l = torch.mean((output - target) ** 2)
#         negative_penalization = torch.nan_to_num(torch.mean(-1 * penalize_negative * (output[output < 0])))
#         # if negative_penalization != 0:
#         if negative_penalization != 0:
#             print(f"NP = {negative_penalization}, MSE_l ={MSE_l} ")
#
#         loss = MSE_l + negative_penalization
#         return loss
#
#     return my_loss

def create_custom_loss(alpha,beta,MAE=False):
    def custom_loss(output, target_output, hidden_layer_representation=False, target_hidden=None):
        # Compute the standard loss (e.g., mean squared error) for the output layer
        standard_loss = F.mse_loss(output.squeeze(), target_output)
        #print(standard_loss)

        if isinstance(hidden_layer_representation,torch.Tensor):
            # Compute a loss term based on the hidden layer representation and its target
            hidden_loss = F.mse_loss(hidden_layer_representation, target_hidden)
            # print(hidden_loss.shape)
        elif hidden_layer_representation == False:
            hidden_loss = 0
        else:
            raise Error("The hidden layer must either be false, or tensor")

        # Combine the two loss terms with a weighting factor alpha
        total_loss = beta * standard_loss + alpha * hidden_loss

        return total_loss

    def custom_loss_MAE(output, target_output, hidden_layer_representation=False, target_hidden=None):
        # Compute the standard loss (e.g., mean squared error) for the output layer
        standard_loss = F.l1_loss(output.squeeze(), target_output)

        # Compute a loss term based on the hidden layer representation and its target
        if isinstance(hidden_layer_representation, torch.Tensor):
            # Compute a loss term based on the hidden layer representation and its target
            hidden_loss = F.l1_loss(hidden_layer_representation, target_hidden)
            # print(hidden_loss.shape)
        elif hidden_layer_representation == False:
            hidden_loss = 0
        else:
            raise Error("The hidden layer must either be false, or tensor")
        # Combine the two loss terms with a weighting factor alpha
        total_loss = beta * standard_loss + alpha * hidden_loss

        return total_loss

    if not MAE:
        loss = custom_loss
    else:
        loss = custom_loss_MAE
    return loss


# def custom_loss(output, target_output, hidden_layer_representation, target_hidden, alpha=0.0,beta = 1):
#     # Compute the standard loss (e.g., mean squared error) for the output layer
#     standard_loss = F.mse_loss(output.squeeze(), target_output)
#
#     # Compute a loss term based on the hidden layer representation and its target
#     hidden_loss = F.mse_loss(hidden_layer_representation, target_hidden)
#
#     # Combine the two loss terms with a weighting factor alpha
#     total_loss = beta*standard_loss + alpha * hidden_loss
#
#     return total_loss

# def custom_loss_MAE(output, target_output, hidden_layer_representation, target_hidden, alpha=0.0,beta = 1):
#     # Compute the standard loss (e.g., mean squared error) for the output layer
#     standard_loss = torch.nn.L1Loss(output.squeeze(), target_output)
#
#     # Compute a loss term based on the hidden layer representation and its target
#     hidden_loss = torch.nn.L1Loss(hidden_layer_representation, target_hidden)
#
#     # Combine the two loss terms with a weighting factor alpha
#     total_loss = beta*standard_loss + alpha * hidden_loss
#
#     return total_loss

# def train_and_get_loss(model,tr_in,tr_out,nb_epochs,lr,print_ = False):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     for epoch in range(nb_epochs):
#         model.train()
#         optimizer.zero_grad()
#         # Forward pass
#         train_predictions = model(tr_in.float())
#         train_loss = torch.nn.MSELoss()(train_predictions.float().squeeze(), tr_out.float())
#
#         # Backward pass
#         # optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#         #Print the training loss every 10 epochs
#         if print_ and (epoch + 1) % 10 == 0:
#             print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}')
#     train_predictions = model(tr_in.float())
#     train_loss = torch.nn.MSELoss()(train_predictions.float().squeeze(), tr_out.float())
#     return train_loss
