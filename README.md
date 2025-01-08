# GNN Jamming Source Localization

## Overview
This project addresses the problem of jamming source localization, modeled as a graph-based regression problem. Each drone's data sample contributes to a node in graph \( G = (V, E) \), with vertices \( V \) representing the drones and edges \( E \) defined by the \( k \)-nearest neighbors based on Euclidean distances between nodes. This approach emphasizes the relational positional information and the noise signal reception of each node, leveraging Graph Neural Networks (GNNs) for accurate localization. We use a Graph Attention Network (GAT) to dynamically weigh the features of neighboring nodes, enhancing model sensitivity to the most informative parts of the data.

## Model Architecture

### GAT Layers
- **GAT Layers**: Multiple layers of graph attention are used to update node features dynamically.
- **Attention Mechanisms**: Each layer computes attention coefficients for pairs of nodes, focusing on significant relational data.
- **Feature Aggregation**: Aggregates neighbor features weighted by attention coefficients independently for each attention head.
- **Multi-Head Attention**: Employs multiple attention mechanisms to improve learning stability and capacity, with outputs concatenated for further processing.

### Output Layer
- The final feature representation is regressed to predict the jammer’s exact coordinates.
- Dropout is applied for regularization, and non-linearities (LeakyReLU) are used between layers.

## Repository Structure
```
gnn-jamming-source-localization/
│
├── main.py                     # Orchestrates the training and evaluation processes.
├── train.py                    # Contains the training loop, model initialization, and evaluation functions.
├── data_processing.py          # Prepares and preprocesses data for model input.
├── model.py                    # Defines the Graph Attention Network architecture.
├── utils.py                    # Utility functions for various tasks throughout the project.
├── custom_logging.py           # Custom logging configurations for enhanced readability.
├── inference.py                # Inference script for testing the model with sample data. 
└── config.py                   # Configuration parameters for the model and training process.
```

## Installation
To install the necessary Python libraries for this project, execute the following command:
```bash
pip install -r requirements.txt
```
Ensure Python 3.8 or more recent version is installed on your system.

## Usage
Run the following command to run inference on simulation data. :
```bash
python inference.py
```