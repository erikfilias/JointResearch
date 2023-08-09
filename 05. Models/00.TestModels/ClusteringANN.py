# Define the autoencoder architecture
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Suppose you have an array of values
values = np.array([14, 53, 12, 92, 14, 64, 16, 31, 62, 14, 70, 60, 39, 79, 9, 19, 86, 25, 86, 16])

# Normalize the values to a range between 0 and 1
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(values.reshape(-1, 1))

# Convert the data to PyTorch tensors
data = torch.from_numpy(normalized_values).float()

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
    outputs = autoencoder(data)
    loss = criterion(outputs, data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get the encoded representations from the Autoencoder
encoded_data = autoencoder.encoder(data).detach().numpy()

# Apply K-means clustering to the encoded data
kmeans = KMeans(n_clusters=3)  # Specify the desired number of clusters
kmeans.fit(encoded_data)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Group the original values based on the cluster labels
grouped_values = {}
for i, value in enumerate(values):
    label = cluster_labels[i]
    if label not in grouped_values:
        grouped_values[label] = []
    grouped_values[label].append(value)

# Print the grouped values
for group, values in grouped_values.items():
    print(f"Group {group + 1}: {values}")
