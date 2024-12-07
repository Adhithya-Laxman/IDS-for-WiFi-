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
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


class DBN:
    def __init__(self, input_size, layers, mode = 'bernoulli', gpu = False, k = 5, savefile = None):
        # K = ? 
        self.layers = layers #RBM Layers -- it stores a list with each element specifying the number of hidden units in each RBM layer
        self.input_size = input_size #Batch input size to DBN
        self.layer_parameters = [{'W' : None, 'hb' : None, 'vb' : None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile

    # v, h -- same for both RBM and DBN

    def sample_v(self, y, W, vb):
        wy = torch.mm(y, W)
        activation = wy + vb
        p_v_given_h =torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        else:
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))

    def sample_h(self, x, W, hb):
        wx = torch.mm(x, W.t())
        activation = wx + hb
        p_h_given_v = torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        else:
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))
    '''
    Stacking and Averaging: After generating k versions of x_dash, they are stacked together using torch.stack(x_gen) and then averaged with torch.mean(x_dash, dim=0). The 
    resulting x_dash is a smoother or more general representation of the input to the next layer, which may help stabilize training or reduce the noise from the sampling process.


    '''
    '''
    In your generate_input_for_layer function, after repeatedly sampling inputs using the sample_h function, the code stacks and averages the samples. Let’s break down what's happening with an example.

    1. Sampling Multiple Times (self.k samples):
    You sample the input x multiple times (self.k times) using the sample_h function, which outputs x_dash for each iteration. This process is repeated for self.k times to generate k different versions of x_dash, which are then stored in the list x_gen.

    2. Stacking the Samples:
    After k samples are generated and stored in x_gen, the following line:

    python
    Copy code
    x_dash = torch.stack(x_gen)
    Explanation:

    torch.stack(x_gen) combines the list of tensors (x_gen) into a single tensor. Each tensor in x_gen (i.e., each x_dash) has the same shape, say (n, m), where n is the batch size (or number of data points) and m is the number of features (or units).
    After stacking, x_dash becomes a 3D tensor of shape (k, n, m), where k is the number of samples, n is the batch size, and m is the number of features.
    3. Averaging the Stacked Samples:
    Next, you average the stacked samples along the first dimension (i.e., the dimension corresponding to k samples):

    python
    Copy code
    x_dash = torch.mean(x_dash, dim=0)
    Explanation:

    This takes the mean across the k samples for each element in the (n, m) tensor.
    The result is a 2D tensor of shape (n, m), where each value in the new x_dash is the average of the corresponding values across the k samples.
    This averaging process is used to smooth the input by reducing the variance introduced by individual samples. The resulting x_dash is less noisy and more robust before being passed to the next layer.

    Example with Shapes:
    Let's assume:

    x is of shape (2, 3) (batch size 2, features 3).
    self.k = 3 (3 samples).
    For each iteration, sample_h generates a new version of x_dash. After 3 iterations, x_gen will contain 3 tensors, each of shape (2, 3).

    torch.stack(x_gen) will result in a 3D tensor of shape (3, 2, 3):

    lua
    Copy code
    [[[x11_1, x12_1, x13_1],  -> First sample
    [x21_1, x22_1, x23_1]],

    [[x11_2, x12_2, x13_2],  -> Second sample
    [x21_2, x22_2, x23_2]],

    [[x11_3, x12_3, x13_3],  -> Third sample
    [x21_3, x22_3, x23_3]]]
    torch.mean(x_dash, dim=0) will compute the mean across the first dimension (sample dimension), resulting in a 2D tensor of shape (2, 3):

    lua
    Copy code
    [[mean(x11_1, x11_2, x11_3), mean(x12_1, x12_2, x12_3), mean(x13_1, x13_2, x13_3)],
    [mean(x21_1, x21_2, x21_3), mean(x22_1, x22_2, x22_3), mean(x23_1, x23_2, x23_3)]]
    This averaging ensures the final input x_dash to the next layer is a smoothed version, reducing the effects of noise from individual sampling steps.
    '''
    def generate_input_for_layer(self, layer_number, x):
        # x -- input to the layer
        if layer_number > 0: 
            x_gen = []
            for _ in range(self.k):
                x_dash = x.clone()
                for i in range(layer_number):
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
                
                x_gen.append(x_dash)
            
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim = 0)
        else:
            x_dash = x.clone()
        return x_dash

    def train_DBN(self, x):
        # x -- input to the DBN (Dataset)

        for index, layer in enumerate(self.layers):
            # Find the number of visible units for current RBM Layer
            if index == 0:
                vn = self.input_size #If it it the first RBM give all the inputs (154 components) as the number of visible units vn
            else:
                vn = self.layers[index - 1] #Else the number of visible units = number of hidden units in the previous RBM layer

            hn = self.layers[index] #Initialize number of hidden units for the current RBM layer

            # Generate and train an RBM for this layer
            
            rbm = RBM(n_visible = vn, n_hidden = hn, mode='bernoulli',lr=0.0005, k=10, batch_size=128, gpu=False, optimizer='adam', early_stopping_patience=10, epochs=20)
            # Generate input layers for current RBM layer
            x_dash = self.generate_input_for_layer(index, x=x)
            # Train the RBM
            rbm.train(x_dash)
            # Update the layer parameters after training
            self.layer_parameters[index]['W'] = rbm.W.cpu()
            self.layer_parameters[index]['hb'] = rbm.hb.cpu()
            self.layer_parameters[index]['vb'] = rbm.vb.cpu()

            print("FINISHED TRAINING LAYER: ", index, "TO", index+1)

        if self.savefile is not None:
            torch.save(self.layer_parameters, self.savefile)
    
    def initialize_dbn(self, model_path):
        """
        Load a pretrained DBN model from a file and initialize the DBN with the weights and biases.

        Args:
            model_path (str): Path to the saved DBN-RBM model file.
        """
        print(f"Loading pretrained DBN model from: {model_path}")
        try:
            # Load the saved model state dictionary
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Extract layer weights and biases
            for i, layer in enumerate(self.layer_parameters):
                layer['W'] = checkpoint[f'layer_{i}_W']
                layer['hb'] = checkpoint[f'layer_{i}_hb']
                layer['vb'] = checkpoint[f'layer_{i}_vb']
            
            print("Pretrained DBN model successfully loaded and initialized.")
        except Exception as e:
            print(f"Error loading DBN model: {e}")
            raise


    def reconstructor(self, x):
        '''
            A dummy function jus ton display the hidden and regenerated unit at each step
        '''
        x_gen = []  
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(len(self.layer_parameters)):
                _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(len(self.layer_parameters)):
                i = len(self.layer_parameters)-1-i
                _, y_dash = self.sample_v(y_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['vb'])
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)

        return y_dash, x_dash
    
    def initialize_model(self):
        # Construct a FNN from the pretrained RBM weights and return the neural network
        print("The Last layer will not be activated. The rest are activated using the Sigmoid Function")

        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer['W'].shape[1], layer['W'].shape[0]))
            if index < len(self.layer_parameters) - 1:
                modules.append(torch.nn.Sigmoid())
        
        model = torch.nn.Sequential(*modules)
        for layer_no, layer in enumerate(model):
            if layer_no // 2 == len(self.layer_parameters) - 1:
                break
            if layer_no%2 == 0:
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]['W'])
                model[layer_no].bias = torch.nn.Parameter(self.layer_parameters[layer_no//2]['hb'])

        return model


def trial_dataset():
	dataset = []
	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				t.append(0)
			else:
				t.append(1)
		dataset.append(t)

	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				t.append(1)
			else:
				t.append(0)
		dataset.append(t)

	dataset = np.array(dataset, dtype=np.float32)
	np.random.shuffle(dataset)
	dataset = torch.from_numpy(dataset)
	return dataset


def train_FNN(model, dataloader, num_epochs=10):
    criterion = torch.nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    all_labels = []  # To store true labels
    all_probs = []   # To store predicted probabilities

    for epoch in range(num_epochs):
        total, correct = 0, 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate predictions and accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append labels and probabilities for ROC curve
            all_labels.extend(labels.cpu().numpy())  # True labels
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())  # Predicted probabilities

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    # After training, calculate the ROC curve
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # For binary classification, you can use the probabilities for class 1 (if using softmax)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])  # Assuming class 1 is positive
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



    # def train_classifier(model, dataset, labels, epochs=100, batch_size=128):
    #     criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for softmax
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    #     for epoch in range(epochs):
    #         model.train()
    #         for i in range(0, len(dataset), batch_size):
    #             batch_x = dataset[i:i + batch_size]
    #             batch_y = labels[i:i + batch_size]

    #             optimizer.zero_grad()  # Zero gradients
    #             outputs = model(batch_x)  # Forward pass
    #             loss = criterion(outputs, batch_y)  # Compute loss
    #             loss.backward()  # Backward pass
    #             optimizer.step()  # Update weights

    #         if (epoch + 1) % 10 == 0:
    #             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')




import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate(model, train_loader, test_loader, num_epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store metrics
    train_accuracies, test_accuracies = [], []
    train_losses = []  # To store training losses for learning curve
    precisions, recalls, f1_scores, detection_rates = [], [], [], []
    all_labels, all_predictions, all_probs = [], [], []  # For confusion matrix and ROC curve

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct, train_total = 0, 0
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)
        train_losses.append(epoch_loss / len(train_loader))  # Average loss

        # Testing phase
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Store predictions and labels for metric calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # Store predicted probabilities for ROC curve
                probs = torch.nn.Softmax(dim=1)(outputs)
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        test_accuracy = 100 * test_correct / test_total
        precision = precision_score(all_labels, all_predictions, average="macro") * 100
        recall = recall_score(all_labels, all_predictions, average="macro") * 100
        f1 = f1_score(all_labels, all_predictions, average="macro") * 100
        detection_rate = recall  # Detection rate is same as recall in IDS context

        # Append metrics for graphing
        test_accuracies.append(test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        detection_rates.append(detection_rate)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%, "
              f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, "
              f"F1-Score: {f1:.2f}%, Detection Rate: {detection_rate:.2f}%")    # Plot graphs

    # 1. Line Graph: Accuracy vs Epochs (Training vs Testing)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', linestyle='-', color='green')
    plt.title('Accuracy vs Epochs', fontsize=16, weight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Line Graph: Precision, Recall, F1-Score, Detection Rate vs Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, precisions, label='Precision', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, recalls, label='Recall', marker='o', linestyle='-', color='green')
    plt.plot(epochs, f1_scores, label='F1-Score', marker='o', linestyle='-', color='orange')
    plt.plot(epochs, detection_rates, label='Detection Rate', marker='o', linestyle='-', color='red')
    plt.title('Metrics vs Epochs', fontsize=16, weight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. Bar Graph: Comparison of Final Epoch Metrics (Precision, Recall, F1, Test Accuracy)
    final_metrics = [precisions[-1], recalls[-1], f1_scores[-1], test_accuracies[-1]]
    metrics_labels = ['Precision', 'Recall', 'F1-Score', 'Test Accuracy']
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_labels, final_metrics, color=['blue', 'green', 'orange', 'red'])
    plt.title('Comparison of Final Epoch Metrics', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('Metrics', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 4. Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 5. ROC Curve
    # Binarize the true labels (since ROC curve works with binary/multi-class labels)
    y_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])  # Adjust classes as needed

    # Compute ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(y_bin.ravel(), np.array(all_probs).ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # 6. Boxplot: Accuracy Distribution over Epochs
    plt.figure(figsize=(10, 6))
    plt.boxplot([train_accuracies, test_accuracies], labels=['Train Accuracy', 'Test Accuracy'])
    plt.title('Accuracy Distribution over Epochs', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.tight_layout()
    plt.show()
    plot_metrics_and_confusion_matrix(
    train_accuracies, test_accuracies, 
    precisions, recalls, f1_scores, 
    all_labels, all_predictions
    )

    return model


def plot_metrics_and_confusion_matrix(train_accuracies, test_accuracies, precisions, recalls, f1_scores, all_labels, all_predictions):
    # Create a 1x3 subplot layout (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot accuracies
    axes[0].plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy', color='blue')
    axes[0].plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy', color='red')
    axes[0].set_title('Accuracy vs Epoch', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy (%)', fontsize=14)
    axes[0].legend(loc='lower right')

    # Plot metrics (Precision, Recall, F1 Score)
    axes[1].plot(range(len(precisions)), precisions, label='Precision', color='green')
    axes[1].plot(range(len(recalls)), recalls, label='Recall', color='blue')
    axes[1].plot(range(len(f1_scores)), f1_scores, label='F1 Score', color='orange')
    axes[1].set_title('Metrics vs Epoch', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Percentage (%)', fontsize=14)
    axes[1].legend(loc='lower right')

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[2], annot_kws={"size": 14})
    axes[2].set_title('Confusion Matrix', fontsize=16)
    axes[2].set_xlabel('Predicted Label', fontsize=14)
    axes[2].set_ylabel('True Label', fontsize=14)

    plt.tight_layout()
    plt.show()

# Training function with manual backpropagation and weight update
def train_classifier_manual(model, dataset, labels, epochs=100, batch_size=128, learning_rate=0.001):
    def cross_entropy_loss(outputs, targets):
        """
        Compute cross-entropy loss manually.
        """
        # Apply softmax to outputs
        softmax_outputs = torch.exp(outputs) / torch.exp(outputs).sum(dim=1, keepdim=True)
        
        # Select the probabilities corresponding to the target labels
        target_probs = softmax_outputs[range(len(targets)), targets]
        
        # Calculate negative log-likelihood
        log_probs = -torch.log(target_probs + 1e-10)
        
        # Return mean loss
        return log_probs.mean()
    
    def compute_gradients_and_update_weights(model, inputs, outputs, targets):
        """
        Perform backpropagation manually and update weights.
        """
        # Apply softmax for predictions
        softmax_outputs = torch.exp(outputs) / torch.exp(outputs).sum(dim=1, keepdim=True)
        
        # Calculate the gradient of loss with respect to outputs
        grad_outputs = softmax_outputs
        grad_outputs[range(len(targets)), targets] -= 1
        grad_outputs /= len(targets)  # Average over batch size

        # Backpropagation manually for each layer in the model
        for layer in reversed(model):  # Assuming model layers are in a list
            if isinstance(layer, torch.nn.Linear):
                # Gradients for weights and biases
                grad_weights = layer.input.T @ grad_outputs  # Weight gradient
                grad_bias = grad_outputs.sum(dim=0)  # Bias gradient
                
                # Update weights and biases
                layer.weight.data -= learning_rate * grad_weights
                layer.bias.data -= learning_rate * grad_bias
                
                # Update gradient for the next layer (backpropagating)
                grad_outputs = grad_outputs @ layer.weight.data.T
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        
        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            batch_x = dataset[i:i + batch_size]
            batch_y = labels[i:i + batch_size]

            # Forward pass
            layer_input = batch_x
            for layer in model:
                if isinstance(layer, torch.nn.Linear):
                    layer.input = layer_input  # Store input for backpropagation
                    layer_output = layer_input @ layer.weight.T + layer.bias
                    layer_input = layer_output
            
            # Final output from the last layer
            outputs = layer_output

            # Calculate loss
            loss = cross_entropy_loss(outputs, batch_y)
            epoch_loss += loss.item()

            # Backpropagation and weight update
            compute_gradients_and_update_weights(model, batch_x, outputs, batch_y)
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (len(dataset) // batch_size):.4f}')


import matplotlib.pyplot as plt
import networkx as nx

# Function to plot the RBM layers
def plot_rbm_layers(dbn):
    fig, axes = plt.subplots(len(dbn.layers), 1, figsize=(5, 10))
    for i, ax in enumerate(axes):
        ax.set_title(f"RBM Layer {i+1}: {dbn.layers[i]} Hidden Units")
        ax.imshow(dbn.layer_parameters[i]['W'].cpu().detach().numpy(), aspect='auto', cmap='viridis')
        ax.set_xlabel('Visible Units')
        ax.set_ylabel('Hidden Units')
    plt.tight_layout()
    plt.show()

# Function to plot the FNN model
def plot_fnn_model(fnn_model):
    graph = nx.DiGraph()

    # Add nodes for each layer in the FNN
    for i, layer in enumerate(fnn_model):
        if isinstance(layer, torch.nn.Linear):
            graph.add_node(i, label=f"Layer {i}: {layer.in_features} -> {layer.out_features}")
            if i > 0:
                graph.add_edge(i - 1, i)  # Connect previous layer to the current one

    # Draw the graph
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title("FNN Architecture")
    plt.show()

# Assuming DBN and FNN are already defined and trained
# After training the DBN, you can visualize the RBM layers and the FNN as follows:

# Create a DBN instance and train it (this part is already in your code)
# dbn = DBN(input_size=154, layers=[100, 50, 25])
# dbn.train_DBN(dataset)  # Pass your dataset here


from torchviz import make_dot




if __name__ == '__main__':
    # Load dataset

    
    # dataset = pd.read_csv(r'C:\Users\Admin\Desktop\2105001\IDS Project\IDS-for-WiFi-\datasets\processed\Preprocessed_set2_10000.csv').astype('float32')
    dataset = pd.read_csv('/home/adminroot/Desktop/2105001/IDS/IDS-for-WiFi-/datasets/processed/Preprocessed_set2_10000.csv').astype('float32')

    features = dataset.iloc[:, :-1].to_numpy()  # All columns except last
    labels = dataset.iloc[:, -1].to_numpy()  # Last column as labels

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()

    # Create DataLoader for training and testing
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Initialize DBN and model
    layers = [148, 128, 64, 32, 16, 8, 4]
    dbn = DBN(148, layers)
    dbn.train_DBN(X_train)  # Assuming DBN pretraining is implemented
    model = dbn.initialize_model()
    print("FINAL FNN MODEL TRAINING...")
    # Train the model with both train and test accuracy tracking
    train_and_evaluate(model, train_loader, test_loader, num_epochs=10)
    # train_classifier_manual(model, train_loader, test_loader, num_epochs=10)
    # Plot the RBM layers
    # plot_rbm_layers(dbn)

    save_path = '/home/adminroot/Desktop/2105001/IDS/IDS-for-WiFi-/src/trained models/dbn_rbm_model_CLIENT2.pth'
    print("States: ", model.state_dict().keys())
    torch.save({
        'model_state_dict': model.state_dict(),  # Save the model's parameters
        'dbn_layers': layers,                   # Save the DBN architecture
        'input_size': 148,                      # Save the input size for reconstruction
        'visible_layer_biases': [layer['vb'] for layer in dbn.layer_parameters]  # Save all visible layer biases (vb) for each layer
    }, save_path)
    
    print(f"Trained DBN-RBM model saved at: {save_path}")

    