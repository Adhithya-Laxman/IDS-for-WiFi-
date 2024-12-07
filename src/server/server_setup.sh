#!/bin/bash

# Set environment variables
cd "$HOME/IDS-for-WiFi-/"
ENV_FILE="$HOME/.my_env_vars"
echo "Setting environment variables..."
cat <<EOL > "$ENV_FILE"
export PARAMS_PATH="~/IDS-for-WiFi-/src/trained models/GLOBAL_dbn_rbm_model.pth"
export JSON_PATH="~/IDS-for-WiFi-/src/FEDavg/client_configuration.json"
EOL

# Source the environment variables
echo "Sourcing environment variables..."
echo "source $ENV_FILE" >> "$HOME/.bashrc"
source "$ENV_FILE"
