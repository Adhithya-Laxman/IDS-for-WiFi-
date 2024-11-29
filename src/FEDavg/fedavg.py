import json
import torch
import os
from datetime import datetime

class FedAvgUpdater:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.network_summary = self._load_json()
        self.global_model_path = self.network_summary["network_summary"]["last_iteration_summary"]["global_model_path"]
        self.global_model = self._load_global_model()

    def _load_json(self):
        with open(self.json_file_path, "r") as file:
            return json.load(file)

    def _load_global_model(self):
        try:
            return torch.load(self.global_model_path)
        except FileNotFoundError:
            raise Exception(f"Global model not found at {self.global_model_path}!")

    def _load_client_model(self, model_path):
        try:
            return torch.load(model_path)
        except FileNotFoundError:
            raise Exception(f"Client model not found at {model_path}!")

    def display_model_parameters(self, model_path):
        """
        Display the parameters or keys in the loaded model.

        :param model_path: Path to the model file.
        """
        model = self._load_client_model(model_path)
        print(f"Keys in model at {model_path}:")
        for key in model.keys():
            print(f"- {key}")
        print("=" * 40)

        # If the model contains a model_state_dict, display its keys
        if "model_state_dict" in model:
            print("Parameters in model_state_dict:")
            for param_key in model["model_state_dict"].keys():
                print(f"- {param_key}")
            print("=" * 40)

    def _weighted_average(self, client_updates, weights, total_instances):
        """
        Compute the weighted average of client updates using the formula:
        Wt+1 = summation(k=1 to K) (nk/n) * (wk,t+1)

        :param client_updates: List of model model_state_dicts from clients.
        :param weights: List of instances trained by each client in the last iteration.
        :param total_instances: Total instances trained across all clients.
        :return: Weighted average of model parameters.
        """
        global_update = {key: torch.zeros_like(value) for key, value in self.global_model["model_state_dict"].items()}

        # Compute the weighted average update
        for i, client_update in enumerate(client_updates):
            weight_factor = weights[i] / total_instances  # (nk/n)
            for key in global_update.keys():
                if key in client_update:  # Update only if the key exists in both models
                    global_update[key] += client_update[key] * weight_factor

        return global_update

    def update_global_model(self):
        """Update the global model using FedAvg."""
        clients = self.network_summary["network_summary"]["clients"]
        client_updates = []
        weights = []
        total_instances = 0

        # Collect client updates and weights
        for client in clients:
            model_path = client["model_path"]
            instances_trained = client["instances_trained_last_iteration"]

            # Load client model and calculate weight
            client_model = self._load_client_model(model_path)
            if "model_state_dict" not in client_model:
                raise ValueError(f"Client model at {model_path} does not contain a 'model_state_dict'.")
            
            client_updates.append(client_model["model_state_dict"])
            weights.append(instances_trained)
            total_instances += instances_trained

        # Perform FedAvg
        global_update = self._weighted_average(client_updates, weights, total_instances)

        # Update the global model parameters
        self.global_model["model_state_dict"].update(global_update)

        # Save the updated global model
        torch.save(self.global_model, self.global_model_path)
        print(f"Global model updated and saved to {self.global_model_path}")

    def display_model_parameters(self, model_path):
        """
        Display the parameters (keys) and their corresponding value shapes in the loaded model.

        :param model_path: Path to the model file.
        """
        model = self._load_client_model(model_path)
        print(f"Keys and Value Shapes in Model at {model_path}:")
        print("=" * 40)
        
        for key, value in model.items():
            if hasattr(value, "shape"):  # For tensors or arrays
                print(f"{key}: Shape = {value.shape}")
            else:
                print(f"{key}: Type = {type(value)}")
        print("=" * 40)

        # If the model contains a model_state_dict, display its keys and their shapes
        if "model_state_dict" in model:
            print("Parameters in model_state_dict and their shapes:")
            for param_key, param_value in model["model_state_dict"].items():
                if hasattr(param_value, "shape"):
                    print(f"{param_key}: Shape = {param_value.shape}")
                else:
                    print(f"{param_key}: Type = {type(param_value)}")
            print("=" * 40)

import json
import torch
from datetime import datetime
from fedavg import FedAvgUpdater  # Assuming the class is in fedavg.py

def main():
    # Path to the network configuration JSON file
    json_file_path = r"C:\Users\Admin\Desktop\2105001\IDS Project\IDS-for-WiFi-\src\FEDavg\client_configuration.json"  # Update the path as needed

    # Initialize FedAvgUpdater with the configuration file
    updater = FedAvgUpdater(json_file_path)

    # Display the parameters of the global model
    print("Displaying Global Model Parameters:")
    updater.display_model_parameters(updater.global_model_path)

    # Optionally, display the parameters for each client model
    clients = updater.network_summary["network_summary"]["clients"]
    for client in clients:
        print(f"\nDisplaying parameters for client {client['name']} (ID: {client['client_id']}):")
        updater.display_model_parameters(client["model_path"])

    # Update the global model using FedAvg
    print("\nUpdating Global Model using FedAvg...")
    updater.update_global_model()

    # Optionally, display the updated global model parameters
    print("\nDisplaying Updated Global Model Parameters:")
    updater.display_model_parameters(updater.global_model_path)

if __name__ == "__main__":
    # Run the main function
    main()
