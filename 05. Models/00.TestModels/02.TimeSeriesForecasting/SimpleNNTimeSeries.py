# =================================================================
# libraries
# =================================================================
import torch
import altair as alt
import numpy  as np
import pandas as pd
# =================================================================
# Functions & classes
# =================================================================
# Split a univariate sequence into samples
def split_sequenceUStep(sequence, n_steps_in):
    X, y = list(), list()

    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        # Check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break

        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

# Define the neural network model
class ElectricityDemandForecasting(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size)
        # Self.hidden_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # Define the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        hidden = torch.relu(self.hidden_layer1(input))
        # Hidden = torch.relu(self.hidden_layer2(hidden))
        output = self.output_layer(hidden)
        return output

# =================================================================
# main function
# =================================================================
def main(n_steps_in, n_model, n_neurons):
    # Load the data
    data = pd.read_csv('electricity_demand.csv', header=0, index_col=0)

    # Prepare the data
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    x1, y1 = split_sequenceUStep(train_data['demand'].values, n_steps_in)
    x2, y2 = split_sequenceUStep(test_data['demand'].values, n_steps_in)

    # Convert the data into PyTorch tensors
    train_inputs = torch.from_numpy(x1)
    train_targets = torch.from_numpy(y1)
    test_inputs = torch.from_numpy(x2)
    test_targets = torch.from_numpy(y2)

    # Initialize the model and optimizer
    model = n_model(input_size=n_steps_in, hidden_size=n_neurons, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(100):
        # Forward pass
        train_predictions = model(train_inputs.float())
        train_loss = torch.nn.MSELoss()(train_predictions.float().squeeze(), train_targets.float())

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print the training loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}')

    # Evaluate the model on the test set
    test_predictions = model(test_inputs.float())
    test_loss = torch.nn.MSELoss()(test_predictions.float().squeeze(), test_targets.float())
    print(f'Test Loss: {test_loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

    # # Load the saved model
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()

    # # Predict demand for new input
    # with torch.no_grad():
    #     predicted_demand = model(new_input)

    # Convert the test predictions and test output to NumPy arrays
    test_targets = test_targets.detach().numpy()
    test_predictions = test_predictions.detach().numpy()

    # Saving the array into pandas dataframe and CSV files
    test_targets_df = pd.DataFrame(test_targets, columns=['Test Demand'])
    test_targets_df.to_csv('test_targets.csv', index=False)
    test_predictions_df = pd.DataFrame(test_predictions, columns=['Predicted Demand'])
    test_predictions_df.to_csv('test_predictions.csv', index=False)

    test_targets_df     = test_targets_df.iloc[:168]
    test_predictions_df = test_predictions_df.iloc[:168]

    # Plotting the forecasting and targe curves
    time_df = pd.DataFrame({'LoadLevel': pd.date_range(start='2023-05-04 00:00:00', periods=len(test_predictions_df), freq='H')})
    frames = [time_df, test_predictions_df, test_targets_df]
    result = pd.concat(frames, axis=1).set_index('LoadLevel')

    source = result.stack().reset_index().rename(columns={'level_1': 'Demand', 0: 'Value'})
    lines = (alt.Chart(source).mark_line().encode(x="LoadLevel", y="Value", color="Demand")).properties(width=1500, height=500)
    lines.save('Plot.html', embed_options={'renderer': 'svg'})

# =============================================================================================================================================================
# script execution
# =============================================================================================================================================================
if __name__ == '__main__':
    # number of previous sample to look for correlations
    step_in = 168
    neurons = 20
    main(n_steps_in=step_in, n_model=ElectricityDemandForecasting, n_neurons=neurons)