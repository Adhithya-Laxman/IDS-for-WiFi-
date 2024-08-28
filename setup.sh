#!/bin/bash

# Project root directory
PROJECT_DIR="."

# Create the main project directory
mkdir -p $PROJECT_DIR

# Create directories for datasets
mkdir -p $PROJECT_DIR/datasets/AWID
mkdir -p $PROJECT_DIR/datasets/processed

# Create directories for models
mkdir -p $PROJECT_DIR/models/dbn_model

# Create directories for notebooks
mkdir -p $PROJECT_DIR/notebooks

# Create source code directories
mkdir -p $PROJECT_DIR/src/server
mkdir -p $PROJECT_DIR/src/clients/client1
mkdir -p $PROJECT_DIR/src/clients/client2
mkdir -p $PROJECT_DIR/src/clients/client3
mkdir -p $PROJECT_DIR/src/models

# Create directories for logs
mkdir -p $PROJECT_DIR/logs

# Create directories for results and graphs
mkdir -p $PROJECT_DIR/results/graphs

# Create directories for configuration files
mkdir -p $PROJECT_DIR/config

# Create directories for scripts
mkdir -p $PROJECT_DIR/scripts

# Create README.md, requirements.txt, and .gitignore
touch $PROJECT_DIR/README.md
touch $PROJECT_DIR/requirements.txt
touch $PROJECT_DIR/.gitignore

# Create placeholder files for configuration
touch $PROJECT_DIR/config/server_config.yaml
touch $PROJECT_DIR/config/client_config.yaml
touch $PROJECT_DIR/config/model_config.yaml

# Create placeholder files for logs
touch $PROJECT_DIR/logs/server.log
touch $PROJECT_DIR/logs/client1.log
touch $PROJECT_DIR/logs/client2.log
touch $PROJECT_DIR/logs/client3.log

# Create placeholder Python files for source code
touch $PROJECT_DIR/src/server/server.py
touch $PROJECT_DIR/src/server/utils.py
touch $PROJECT_DIR/src/server/config.py
touch $PROJECT_DIR/src/clients/client1/client.py
touch $PROJECT_DIR/src/clients/client1/utils.py
touch $PROJECT_DIR/src/clients/client2/client.py
touch $PROJECT_DIR/src/clients/client2/utils.py
touch $PROJECT_DIR/src/clients/client3/client.py
touch $PROJECT_DIR/src/clients/client3/utils.py
touch $PROJECT_DIR/src/models/dbn.py
touch $PROJECT_DIR/src/models/fl_model.py
touch $PROJECT_DIR/src/models/train_centralized.py
touch $PROJECT_DIR/src/models/train_federated.py

# Create placeholder Python files for scripts
touch $PROJECT_DIR/scripts/preprocess_data.py
touch $PROJECT_DIR/scripts/split_data.py

echo "Project structure created successfully in '$PROJECT_DIR'."
