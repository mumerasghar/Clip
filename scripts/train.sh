#!/bin/bash

# Parse arguments.
GPUS=$1
CPUS=$2
CFG=$3
INFO=$4

python train.py --gpus $GPUS --cpus $CPUS --cfg $CFG --info $INFO
