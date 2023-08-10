# Define the libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Hyperparameters
DIR = os.path.dirname(os.path.realpath(__file__))
CASE = '3-bus'

# %% Reading data from CSV files
_path = os.path.join(DIR, CASE)
# reading the admittance matrix
Y = pd.read_csv(_path + '/3.Out' + '/oT_Result_NN_Input_' + CASE + '.csv', index_col=[0, 1, 2, 3])

# selecting the impedance matrix
MatrixYReal = Y.loc[Y['Dataset'] == 'MatrixYReal']
MatrixYImag = Y.loc[Y['Dataset'] == 'MatrixYImag']

MatrixYReal = np.array(MatrixYReal['Value'].values).reshape(-1,1)
MatrixYImag = np.array(MatrixYImag['Value'].values).reshape(-1,1)


# Combine real and imaginary components into complex values
complex_matrix = MatrixYReal + 1j * MatrixYImag

# Flatten the complex matrix for input to the autoencoder
flattened_matrix = complex_matrix.reshape(complex_matrix.shape[0], -1)

# Normalize the data
normalized_data = (flattened_matrix - flattened_matrix.mean()) / flattened_matrix.std()

# Convert to PyTorch tensor
input_data = torch.tensor(normalized_data, dtype=torch.float32)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set the dimensions of the input and encoding
input_dim = 1
encoding_dim = 1

# Create an instance of the Autoencoder model
autoencoder = Autoencoder(input_dim, encoding_dim)

# Train the Autoencoder
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(input_data)
    loss = criterion(outputs, input_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get the encoded representations from the Autoencoder
encoded_data = autoencoder.encoder(input_data).detach().numpy()