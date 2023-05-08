# =================================================================
# libraries
# =================================================================
import torch
import altair as alt
import numpy  as np
import pandas as pd
from SimpleNNTimeSeries import split_sequenceUStep, main_f
# Define the neural network model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size)
        # self.hidden_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.dropout = torch.nn.Dropout(p=0.1)  # dropout probability of 0.5
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # Define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden = torch.relu(self.hidden_layer1(input))
        # hidden = torch.sigmoid(self.hidden_layer2(hidden))
        output = self.output_layer(hidden)
        return output


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(input.view(len(input), batch_size, -1), hidden)
        output = self.linear(lstm_out[-1])
        return output

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_size)
        c0 = torch.zeros(1, batch_size, self.hidden_size)
        return h0, c0


# =================================================================
# main function
# =================================================================
steps = 168
# df = dict()
df = pd.DataFrame(columns=['Neurons', 'Error'])
idx = 0
for i in range(1,200,10):
    print(f'--------------------------> Neurons: {i}')
    loss = main_f(steps,MLP,i)
    df.loc[idx] = [i, loss.item()]
    idx += 1

# Making the Scatter Plot
fig = alt.Chart(df).mark_point().encode(
    # Map the sepalLength to x-axis
    x='Neurons',
    # Map the petalLength to y-axis
    y='Error',
    # y='petalLength',
    # # Map the species to shape
    # shape='species'
)
fig.save('OptimalNumberNeurons.html', embed_options={'renderer': 'svg'})