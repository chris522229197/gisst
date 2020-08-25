#!/bin/bash

# Set up project directories and graph datasets

echo "Setting up directories..."
mkdir test_output
mkdir slurms/output

echo "Processing real-world graphs..."
python3 process_real_graph.py graphs/ mutagenicity
python3 process_real_graph.py graphs/ reddit_binary
python3 process_real_graph.py graphs/ proteins

echo "Done!"
