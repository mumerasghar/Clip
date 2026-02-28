#!/bin/bash

# # Install dependencies.
apt-get update
apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

if [ -d "venv" ]; then
    echo "Found existing virtual environment"
    # Activate the virtual environment and install pip.
    source venv/bin/activate
    echo -e "Using $(python --version) ($(which python))"
else
    # Install Python 3.10.6.
    echo -e "Installing Python 3.10"
    apt update
    apt install software-properties-common -y
    add-apt-repository ppa:deadsnakes/ppa -y
    apt install python3.10 -y
    python3.10 --version

    # Create a virtual environment using Python 3.10
    echo -e "Creating virtual environment"
    apt install python3.10-venv -y
    python3.10 -m venv venv

    # Activate the virtual environment and install pip.
    source venv/bin/activate
    echo -e "Using $(python --version) ($(which python))"

    # Install the required packages.
    echo -e "Installing required packages in $(which python)"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

# Run the training.
start=`date +%s`
SCRIPT=$1
shift 1
ARGS=$@
bash $SCRIPT $ARGS
end=`date +%s`
echo -e "Job took $((end-start)) seconds."

# Deactivate the virtual environment.
deactivate
