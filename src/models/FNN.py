import numpy as np
import torch
import random
from tqdm import trange
from RBM import RBM
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the Feedforward Neural Network (FNN) class
class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(FNN, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))  # Using softmax for multi-class classification

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    


# if __name__ == '__main__': 
#     # Load dataset
#     df = pd.read_csv(r'C:\Users\Admin\Desktop\2105001\IDS Project\IDS-for-WiFi-\datasets\processed\Preprocessed_set1_10000.csv')
#     dataset = torch.from_numpy(df.astype('float32').to_numpy())

#     # Extract features and labels
#     X = dataset[:]  # All columns except the last one
#     y = dataset[:, -1]   # The last column as labels

#     # Convert labels to a format suitable for classification (if necessary)
#     le = LabelEncoder()
#     y = le.fit_transform(y.numpy())  # Encoding labels as integers
#     y = torch.from_numpy(y).long()    # Convert to PyTorch tensor

#     # Train-validate split
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define the DBN architecture
#     layers = [149, 128, 64, 32, 16, 8, 4]
#     dbn = DBN(149, layers)

#     # Pretrain DBN
#     dbn.train_DBN(X_train)

#     # Initialize model from DBN
#     model = dbn.initialize_model()

#     # Perform reconstruction for evaluation (optional)
#     y_reconstructed, _ = dbn.reconstructor(X_train)
#     print('\n\n\n')
#     print("MAE of an all 0 reconstructor:", torch.mean(X_train).item())
#     print("MAE between reconstructed and original sample:", torch.mean(torch.abs(y_reconstructed - X_train)).item())

#     # Now define the FNN for classification
#     fnn_layers = [128, 64, 32, 16]  # Hidden layers for FNN
#     fnn = FNN(input_size=layers[-1], hidden_layers=fnn_layers, output_size=len(le.classes_))

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()  # For multi-class classification
#     optimizer = optim.Adam(fnn.parameters(), lr=0.001)

#     # Training the FNN
#     fnn.train()
#     num_epochs = 100  # Adjust as necessary
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         outputs = fnn(model(X_train))  # Pass through the DBN and then the FNN
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#     # Validate the FNN
#     fnn.eval()
#     with torch.no_grad():
#         val_outputs = fnn(model(X_val))
#         _, predicted = torch.max(val_outputs, 1)
#         accuracy = accuracy_score(y_val.numpy(), predicted.numpy())
#         print(f'Validation Accuracy: {accuracy * 100:.2f}%')


print(6/2*(1+2))