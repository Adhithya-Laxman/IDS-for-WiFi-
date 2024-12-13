import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths
dataset_path = "/home/adminroot/IDS-for-WiFi-/datasets/processed/Preprocessed_set3_10000.csv"
model_path = "/home/adminroot/Desktop/2105001/IDS/IDS-for-WiFi-/src/trained models/GLOBAL_dbn_rbm_model.pth"
model_path = "/home/adminroot/Desktop/2105001/IDS/IDS-for-WiFi-/src/trained models/dbn_rbm_model_CLIENT2.pth"

# Parameters
batch_size = 64  # Batch size for testing

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path)

# Extract features and target
features = data.iloc[:, :-1].values  # All columns except the last one
targets = data.iloc[:, -1].values    # Last column as the target

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.long)  # Use long for classification tasks

# Create DataLoader for testing
test_dataset = TensorDataset(features_tensor, targets_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model components
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)
model_state_dict = checkpoint['model_state_dict']

# Adjust state_dict keys to match model's expected format
adjusted_state_dict = {}
for key, value in model_state_dict.items():
    if "bias" in key:
        adjusted_state_dict[f"network.{key}"] = value.squeeze()  # Remove extra dimension
    else:
        adjusted_state_dict[f"network.{key}"] = value

# Rebuild the FNN dynamically
class ReconstructedFNN(nn.Module):
    def __init__(self, model_state_dict):
        super(ReconstructedFNN, self).__init__()
        layers = []
        prev_layer_size = None

        # Extract layer dimensions from weights
        for key, value in model_state_dict.items():
            if "weight" in key:
                in_features = value.shape[1]  # Number of input features
                out_features = value.shape[0]  # Number of output features

                if prev_layer_size is not None:
                    assert in_features == prev_layer_size, \
                        f"Mismatch in expected layer sizes: {prev_layer_size} != {in_features}"

                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.Sigmoid())  # Add sigmoid activation

                prev_layer_size = out_features

        # Add final softmax activation layer
        layers.pop()  # Remove last sigmoid for final layer
        layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Instantiate the FNN
model = ReconstructedFNN(adjusted_state_dict)
model.load_state_dict(adjusted_state_dict)  # Load adjusted state dict
model.to(device)
model.eval()  # Set to evaluation mode

# Test the model
print("Testing model...")
correct = 0
total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Predicted class
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # Update metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

# Save predictions and targets for further analysis
output_df = pd.DataFrame({
    "Target": all_targets,
    "Prediction": all_predictions
})
output_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to 'test_predictions.csv'.")

# Generate and display confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_targets, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image
plt.show()

print("Confusion matrix saved to 'confusion_matrix.png'.")
