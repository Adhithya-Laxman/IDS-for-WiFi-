#!/bin/bash

# Set environment variables
ENV_FILE="$HOME/.client_vars"
echo "Setting environment variables..."
cat <<EOL > "$ENV_FILE"
export PARAMS_PATH="$HOME/Downloads/Sample.pth"
export UNIQUE_ID="client_001"
export NUMBER_ITR="0"
export NAME="$HOSTNAME"
EOL

# Source the environment variables
echo "Sourcing environment variables..."
echo "source $ENV_FILE" >> "$HOME/.bashrc"
source "$ENV_FILE"

# Define the cron job
CRON_SCHEDULE="0 */3 * * *" # Every 3 hours
SCRIPT_TO_RUN="/path/to/your/script.sh"
LOG_FILE="$HOME/cron_job.log"
chmod +x $HOME"/IDS-for-WiFi-/src/models/DBN_client.py"

echo "Scheduling cron job..."
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $SCRIPT_TO_RUN >> $LOG_FILE 2>&1") | crontab -

echo "Setup complete. Environment variables set and cron job scheduled to run every 3 hours."