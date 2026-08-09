#!/bin/bash
# Usage: bash batch_runner.sh <batch_id> <k_soil> <theta_r> <episodes> <seed>
set -e
cd /workspace/astraeus
nohup python harness.py --episodes "$4" --k-soil "$2" --theta-r "$3" \
  --batch-id "$1" --base-seed "$5" --log /workspace/astraeus/log.csv \
  >> "/workspace/astraeus/batch_$1.out" 2>&1 &
echo "batch $1 running in background; tail -f /workspace/astraeus/batch_$1.out"
