#!/bin/bash
# Retry starting the GCP VM until GPU becomes available.
# Usage: bash scripts/retry_vm.sh

while true; do
  echo "[$(date '+%H:%M:%S')] Attempting to start mimic-gpu..."
  gcloud compute instances start mimic-gpu --zone us-central1-a 2>&1
  if [ $? -eq 0 ]; then
    echo "Instance started successfully at $(date)!"
    # Beep to get your attention (works on most terminals)
    echo -e '\a'
    break
  fi
  echo "Still unavailable. Retrying in 10 minutes..."
  sleep 600
done
