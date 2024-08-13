#!/bin/bash

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install opencv-python
pip install torch
pip install numpy
pip install pyvista
pip install matplotlib
pip install nerf
pip install pytorch-lightning
pip install pytest
pip install scikit-image
pip install scikit-learn
pip install h5py
pip install imageio

echo "Dependencies installed successfully!"
