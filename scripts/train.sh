#!/bin/bash

# Parse arguments.
GPUS=$1
CPUS=$2
CFG=$3
INFO=$4

python main.py --gpus $GPUS --cpus $CPUS --cfg $CFG --info $INFO
