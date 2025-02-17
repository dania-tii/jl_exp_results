import os
import pickle
import hashlib
import json

import pandas as pd
import numpy as np
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset, Dataset
from typing import Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import logging
from utils import cartesian_to_polar
from custom_logging import setup_logging
from config import params

from torch_geometric.utils import to_networkx


setup_logging()


class Instance:
    def __init__(self, row):
        # Initialize attributes from the pandas row and convert appropriate fields to numpy arrays only if not already arrays
        if params['inference']:
            self.node_positions = row['node_positions'] if isinstance(row['node_positions'], np.ndarray) else np.array(row['node_positions'])
            self.node_noise = row['node_noise'] if isinstance(row['node_noise'], np.ndarray) else np.array(row['node_noise'])
            self.jammed_at = row['jammed_at']
        else:
            self.num_samples = row['num_samples']
            self.node_positions = row['node_positions'] if isinstance(row['node_positions'], np.ndarray) else np.array(row['node_positions'])
            self.node_noise = row['node_noise'] if isinstance(row['node_noise'], np.ndarray) else np.array(row['node_noise'])
            self.pl_exp = row['pl_exp']
            self.sigma = row['sigma']
            self.jammer_power = row['jammer_power']
            self.jammer_position = row['jammer_position'] if isinstance(row['jammer_position'], np.ndarray) else np.array(row['jammer_position'])
            self.jammer_gain = row['jammer_gain']
            self.id = row['id']
            self.dataset = row['dataset']
            self.jammed_at = row['jammed_at']
            if 'angle_of_arrival' in params['required_features']:
                self.angle_of_arrival = row['angle_of_arrival'] if isinstance(row['angle_of_arrival'], np.ndarray) else np.array(row['angle_of_arrival'])


    def get_crop(self, start, end):
        if 'angle_of_arrival' in params['required_features']:
            cropped_instance = Instance({
                'num_samples': end - start,
                'node_positions': self.node_positions[start:end],
                'node_noise': self.node_noise[start:end],
                'angle_of_arrival': self.angle_of_arrival[start:end],
                'pl_exp': self.pl_exp,
                'sigma': self.sigma,
                'jammer_power': self.jammer_power,
                'jammer_position': self.jammer_position,
                'jammer_gain': self.jammer_gain,
                'id': self.id,
                'dataset': self.dataset,
                'jammed_at': self.jammed_at  # Jammed index remains the same
            })
        else:
            if params['inference']:
                cropped_instance = Instance({
                    'node_positions': self.node_positions[start:end],
                    'node_noise': self.node_noise[start:end],
                    'jammed_at': self.jammed_at
                })
            else:
                cropped_instance = Instance({
                    'num_samples': end - start,
                    'node_positions': self.node_positions[start:end],
                    'node_noise': self.node_noise[start:end],
                    'pl_exp': self.pl_exp,
                    'sigma': self.sigma,
                    'jammer_power': self.jammer_power,
                    'jammer_position': self.jammer_position,
                    'jammer_gain': self.jammer_gain,
                    'id': self.id,
                    'dataset': self.dataset,
                    'jammed_at': self.jammed_at
                })
        return cropped_instance

    def apply_flip(self):
        # Generate two independent random numbers
        r1, r2 = random.random(), random.random()
        # Apply horizontal flip if r1 is less than 0.5
        if r1 < 0.5:
            self.node_positions[:, 1] = -self.node_positions[:, 1]
            self.jammer_position[1] = -self.jammer_position[1]
        # Apply vertical flip if r2 is less than 0.5
        if r2 < 0.5:
            self.node_positions[:, 0] = -self.node_positions[:, 0]
            self.jammer_position[0] = -self.jammer_position[0]

    def apply_rotation(self, degrees):
        # Mapping degrees to numpy rotation functions
        if degrees == 90:
            self.node_positions = np.dot(self.node_positions, np.array([[0, 1], [-1, 0]]))
            self.jammer_position = np.dot(self.jammer_position, np.array([[0, 1], [-1, 0]]))
        elif degrees == 180:
            self.node_positions = -self.node_positions
            self.jammer_position = -self.jammer_position
        elif degrees == 270:
            self.node_positions = np.dot(self.node_positions, np.array([[0, -1], [1, 0]]))
            self.jammer_position = np.dot(self.jammer_position, np.array([[0, -1], [1, 0]]))


class TemporalGraphDataset(Dataset):
    def __init__(self, data, test=False, dynamic=True, discretization_coeff=0.25):
        self.data = data
        self.test = test  # for test set
        self.dynamic = dynamic
        self.discretization_coeff = discretization_coeff

        if self.test:
            if params['inference']:
                self.samples = [Instance(row) for _, row in data.iterrows()]
                self.precomputed_graphs = [self.precompute_graph(instance) for instance in self.samples]
            else:
                # Precompute the graphs during dataset initialization for the test set
                self.samples = self.expand_samples()
                self.precomputed_graphs = [self.precompute_graph(instance) for instance in self.samples]
        else:
            self.samples = [Instance(row) for _, row in data.iterrows()]

    def expand_samples(self):
        expanded_samples = []
        for _, row in self.data.iterrows():
            lb_end = max(int(row['jammed_at']), min(params['max_nodes'], len(row['node_positions'])))
            # lb_end = max(int(row['jammed_at']), min(10, len(row['node_positions'])))
            ub_end = len(row['node_positions'])
            # lb_end = int((ub_end - lb_end) / 2)

            # Define step size
            if self.discretization_coeff == -1:
                step_size = 1
            elif isinstance(self.discretization_coeff, float):
                step_size = max(1, int(self.discretization_coeff * (ub_end - lb_end)))
            else:
                raise ValueError("Invalid discretization coefficient type")

            # Generate instances for various end points with the step size
            for i in range(lb_end, ub_end + 1, step_size):
                instance = Instance(row).get_crop(0, i)
                instance.perc_completion = i/ub_end
                expanded_samples.append(instance)
        print("len expanded samples: ", len(expanded_samples))
        return expanded_samples

    def precompute_graph(self, instance):
        # Create the graph once and engineer the node features
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)
        return graph

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, start_crop=0):
        if self.test:
            # Return the precomputed graph for test set
            return self.precomputed_graphs[idx]

        # For non-test set, perform random cropping
        instance = self.samples[idx]

        # Check if jammed_at is not NaN and set the lower bound for random selection
        if np.isnan(instance.jammed_at):
            raise ValueError("No jammed instance")
        lb_end = max(int(instance.jammed_at), min(params['max_nodes'], len(instance.node_positions)))
        # lb_end = max(int(instance.jammed_at), min(10, len(instance.node_positions)))
        ub_end = len(instance.node_positions)  # The upper bound is always the length of node_positions
        end = random.randint(lb_end, ub_end)

        if 'crop' in params['aug']:
            instance = instance.get_crop(start_crop, end)

        # Apply flipping
        if 'flip' in params['aug']:
            instance.apply_flip()

        if 'rot' in params['aug']:
            instance.apply_rotation(random.choice([0, 90, 180, 270]))  # Choose randomly among 0, 90, 180, or 270 degrees

        instance.perc_completion = end / ub_end

        # Create and engineer the graph on the fly for training
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)

        return graph


def angle_to_cyclical(positions):
    """
    Convert a list of positions from polar to cyclical coordinates.

    Args:
        positions (list): List of polar coordinates [r, theta, phi] for each point.
                          r is the radial distance,
                          theta is the polar angle from the positive z-axis (colatitude),
                          phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        list: List of cyclical coordinates [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
    """
    transformed_positions = []
    if params['3d']:
        for position in positions:
            r, theta, phi = position
            sin_theta = np.sin(theta)  # Sine of the polar angle
            cos_theta = np.cos(theta)  # Cosine of the polar angle
            sin_phi = np.sin(phi)  # Sine of the azimuthal angle
            cos_phi = np.cos(phi)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta, sin_phi, cos_phi])
    else:
        for position in positions:
            r, theta = position
            sin_theta = np.sin(theta)  # Sine of the azimuthal angle
            cos_theta = np.cos(theta)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta])
    return transformed_positions


def cyclical_to_angular(output):
    """
    Convert cyclical coordinates (sin and cos) back to angular coordinates (theta and phi) using PyTorch.

    Args:
        output (Tensor): Tensor containing [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
                         r is the radial distance,
                         theta is the polar angle from the positive z-axis (colatitude),
                         phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        Tensor: Updated tensor with [r, theta, phi] for each point.
    """
    r = output[:, 0]
    sin_theta = output[:, 1]
    cos_theta = output[:, 2]
    sin_phi = output[:, 3]
    cos_phi = output[:, 4]

    theta = torch.atan2(sin_theta, cos_theta)  # Polar angle calculation from sin and cos
    phi = torch.atan2(sin_phi, cos_phi)  # Azimuthal angle calculation from sin and cos

    return torch.stack([r, theta, phi], dim=1)


def mean_centering(coords):
    """Center coordinates using given or calculated bounds."""
    # Calculate the geometric center
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    return centered_coords, center


def center_coordinates_instance(instance):
    """Center and convert drone and jammer positions using shared bounds for a single instance."""
    # logging.info("Centering coordinates for instance")

    # Center coordinates using the calculated mean
    centered_node_positions, center = mean_centering(instance.node_positions)
    instance.node_positions = centered_node_positions
    instance.node_positions_center = center

    # Center jammer position using the same center
    if not params['inference']:
        centered_jammer_position = instance.jammer_position - center
        instance.jammer_position = centered_jammer_position

    return center


def apply_min_max_normalization_instance(instance):
    """Apply min-max normalization to position and RSSI data for an instance."""
    # logging.info("Applying min-max normalization for instance")

    # Normalize Noise values to range [0, 1]
    min_noise = np.min(instance.node_noise)
    max_noise = np.max(instance.node_noise)
    range_noise = max_noise - min_noise if max_noise != min_noise else 1
    normalized_noise = (instance.node_noise - min_noise) / range_noise
    instance.node_noise = normalized_noise

    # Normalize node positions to range [-1, 1]
    min_coords = np.min(instance.node_positions, axis=0)
    max_coords = np.max(instance.node_positions, axis=0)
    range_coords = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
    normalized_positions = 2 * ((instance.node_positions - min_coords) / range_coords) - 1
    instance.min_coords = min_coords
    instance.max_coords = max_coords
    instance.node_positions = normalized_positions

    # Normalize jammer position similarly
    if not params['inference']:
        jammer_position = 2 * ((instance.jammer_position - min_coords) / range_coords) - 1
        instance.jammer_position = jammer_position


def apply_unit_sphere_normalization(instance):
    """
    Apply unit sphere normalization to position data.

    Parameters:
    data (dict): A dictionary containing 'node_positions', an array of positions.

    Returns:
    tuple: A tuple containing the normalized positions and the maximum radius.
    """
    # logging.info("Applying unit sphere normalization")

    # Extract positions from the current row
    positions = instance.node_positions
    if not params['inference']:
        jammer_position = instance.jammer_position

    # Calculate the maximum radius from the centroid
    max_radius = np.max(np.linalg.norm(positions, axis=1))

    # Check for zero radius to prevent division by zero
    if max_radius == 0:
        raise ValueError("Max radius is zero, normalization cannot be performed.")

    # Normalize the positions uniformly
    normalized_positions = positions / max_radius
    if not params['inference']:
        normalized_jammer_position = jammer_position / max_radius

    instance.node_positions = normalized_positions
    if not params['inference']:
        instance.jammer_position = normalized_jammer_position
    instance.max_radius = max_radius


def convert_data_type(data, load_saved_data):
    if load_saved_data:
        dataset_features = params['required_features'] + ['jammer_position', 'jammed_at', 'jammer_power',  'num_samples',  'sigma', 'jammer_power', 'id']
    else:
        # Convert from str to required data type for specified features
        dataset_features = params['required_features'] + ['jammer_position']
    # Apply conversion to each feature directly
    for feature in dataset_features:
        data[feature] = data[feature].apply(lambda x: safe_convert_list(x, feature))


def add_cyclical_features(data):
    """Convert azimuth angles to cyclical coordinates."""
    data['azimuth_angle'] = data.apply(lambda row: [np.arctan2(pos[1] - row['centroid'][1], pos[0] - row['centroid'][0]) for pos in row['node_positions']], axis=1)
    data['sin_azimuth'] = data['azimuth_angle'].apply(lambda angles: [np.sin(angle) for angle in angles])
    data['cos_azimuth'] = data['azimuth_angle'].apply(lambda angles: [np.cos(angle) for angle in angles])


def calculate_proximity_metric(positions, threshold=0.2):
    """Calculate the number of nearby nodes within a given threshold distance."""
    nbrs = NearestNeighbors(radius=threshold).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions)
    return [len(idx) - 1 for idx in indices]  # subtract 1 to exclude the node itself


def add_proximity_count(data):
    """Add proximity feature based on a threshold distance."""
    data['proximity_count'] = data['node_positions'].apply(
        lambda positions: calculate_proximity_metric(np.array(positions))
    )

# Vectorized
def calculate_noise_statistics(subgraphs, stats_to_compute):
    subgraph = subgraphs[0]
    edge_index = subgraph.edge_index
    node_noises = subgraph.x[:, 2]  # Assuming the noise feature is the third feature

    # Create an adjacency matrix from edge_index
    num_nodes = node_noises.size(0)
    # Create an adjacency matrix from edge_index and include self-loops
    adjacency = torch.zeros(num_nodes, num_nodes, device=node_noises.device)
    adjacency[edge_index[0], edge_index[1]] = 1
    torch.diagonal(adjacency).fill_(1)  # Add self-loops

    # Calculate the sum and count of neighbor noises
    neighbor_sum = torch.mm(adjacency, node_noises.unsqueeze(1)).squeeze()
    neighbor_count = adjacency.sum(1)

    # Avoid division by zero for mean calculation
    neighbor_count = torch.where(neighbor_count == 0, torch.ones_like(neighbor_count), neighbor_count)
    mean_neighbor_noise = neighbor_sum / neighbor_count

    # Standard deviation
    neighbor_variance = torch.mm(adjacency, (node_noises**2).unsqueeze(1)).squeeze() / neighbor_count - (mean_neighbor_noise**2)
    std_noise = torch.sqrt(neighbor_variance)

    # Range: max - min for each node's neighbors
    # Expanding the node noises for comparison using adjacency
    expanded_noises = node_noises.unsqueeze(0).repeat(num_nodes, 1)
    max_noise = torch.where(adjacency == 1, expanded_noises, torch.full_like(expanded_noises, float('-inf'))).max(1).values
    min_noise = torch.where(adjacency == 1, expanded_noises, torch.full_like(expanded_noises, float('inf'))).min(1).values
    range_noise = max_noise - min_noise

    # Replace inf values that appear if a node has no neighbors
    range_noise[range_noise == float('inf')] = 0
    range_noise[range_noise == float('-inf')] = 0

    # Other statistics could be added in a similar batch-processed manner
    noise_stats = {
        'mean_noise': mean_neighbor_noise,
        'std_noise': std_noise,
        'range_noise': range_noise,
    }

    return noise_stats



def add_clustering_coefficients(graphs):
    """
    Compute the clustering coefficient for each node in each graph.

    Args:
        graphs (list): List of NetworkX graph objects.

    Returns:
        list: A list of lists, where each sublist contains the clustering coefficients for nodes in a graph.
    """
    all_graphs_clustering_coeffs = []  # This will hold a list of lists, each sublist for a graph

    for graph in graphs:
        graph_clustering_coeffs = []  # Initialize an empty list for current graph's node clustering coefficients

        if len(graph.nodes()) > 0:
            clustering_coeffs = nx.clustering(graph)
            nodes = list(graph.nodes())

            # Populate the clustering coefficients for each node, maintaining the order
            for node in nodes:
                graph_clustering_coeffs.append(clustering_coeffs[node])
        else:
            graph_clustering_coeffs = []

        all_graphs_clustering_coeffs.append(graph_clustering_coeffs)  # Append the completed list for this graph

    return all_graphs_clustering_coeffs


def dynamic_moving_average(x, max_window_size=10):
    num_nodes = x.size(0)
    window_sizes = torch.clamp(num_nodes - torch.arange(num_nodes), min=1, max=max_window_size)
    averages = torch.zeros_like(x)

    for i in range(num_nodes):
        start = max(i - window_sizes[i] // 2, 0)
        end = min(i + window_sizes[i] // 2 + 1, num_nodes)
        averages[i] = x[start:end].mean(dim=0)

    return averages


def engineer_node_features(subgraph):
    if subgraph.x.size(0) == 0:
        raise ValueError("Empty subgraph encountered")

    new_features = []

    # Calculating centroid
    centroid = torch.mean(subgraph.x[:, :2], dim=0)  # Select only x and y

    if 'dist_to_centroid' in params['additional_features']:
        distances = torch.norm(subgraph.x[:, :2] - centroid, dim=1, keepdim=True)
        # print("dist_to_centroid: ", distances)
        new_features.append(distances)

    if 'sin_azimuth' in params['additional_features']:
        azimuth_angles = torch.atan2(subgraph.x[:, 1] - centroid[1], subgraph.x[:, 0] - centroid[0])
        new_features.append(torch.sin(azimuth_angles).unsqueeze(1))
        new_features.append(torch.cos(azimuth_angles).unsqueeze(1))

    # Graph-based noise stats
    graph_stats = ['mean_noise', 'median_noise', 'std_noise', 'range_noise', 'relative_noise', 'wcl_coefficient']
    noise_stats_to_compute = [stat for stat in graph_stats if stat in params['additional_features']]

    if noise_stats_to_compute:
        noise_stats = calculate_noise_statistics([subgraph], noise_stats_to_compute)

        # Add calculated statistics directly to the features list
        for stat in noise_stats_to_compute:
            if stat in noise_stats:
                new_features.append(noise_stats[stat].unsqueeze(1))

    # Moving Average for node noise with adjusted padding
    if 'moving_avg_noise' in params['additional_features']:
        node_noise = subgraph.x[:, 2]  # noise is at position 2
        moving_avg_noise = dynamic_moving_average(node_noise)
        new_features.append(moving_avg_noise.unsqueeze(1))

    # Example of using dynamic moving average for AoA
    if 'moving_avg_aoa' in params['additional_features']:
        sin_aoa = subgraph.x[:, 3]  # sin(AoA) is at position 3
        cos_aoa = subgraph.x[:, 4]  # cos(AoA) is at position 4

        smoothed_sin = dynamic_moving_average(sin_aoa)
        smoothed_cos = dynamic_moving_average(cos_aoa)

        new_features.append(smoothed_sin.unsqueeze(1))
        new_features.append(smoothed_cos.unsqueeze(1))

    if new_features:
        try:
            new_features_tensor = torch.cat(new_features, dim=1)
            subgraph.x = torch.cat((subgraph.x, new_features_tensor), dim=1)
        except RuntimeError as e:
            raise e

    return subgraph


def convert_to_polar(data):
    data['polar_coordinates'] = data['node_positions'].apply(cartesian_to_polar)
    data['polar_coordinates'] = data['polar_coordinates'].apply(angle_to_cyclical)


def polar_to_cartesian(data):
    """
    Convert polar coordinates to Cartesian coordinates using only PyTorch operations.

    Args:
        data (Tensor): Tensor on the appropriate device (GPU or CPU) containing
                       [r, theta, phi] for each point.
                       r is the radial distance,
                       theta is the polar angle from the positive z-axis (colatitude),
                       phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        Tensor: Updated tensor with [x, y, z] for each point.
    """
    r = data[:, 0]
    theta = data[:, 1]  # Polar angle (colatitude)

    if params['3d']:
        phi = data[:, 2]  # Azimuthal angle
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        cartesian_coords = torch.stack([x, y, z], dim=1)
    else:
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        cartesian_coords = torch.stack([x, y], dim=1)

    return cartesian_coords


def convert_output_eval(output, data_batch, data_type, device):
    """
    Convert and evaluate the output coordinates by uncentering them using the stored midpoints.

    Args:
        output (torch.Tensor): The model output tensor.
        data_batch (torch.Tensor): Data batch.
        data_type (str): The type of data, either 'prediction' or 'target'.
        device (torch.device): The device on which the computation is performed.

    Returns:
        torch.Tensor: The converted coordinates after uncentering.
    """
    output = output.to(device)  # Ensure the output tensor is on the right device

    if params['norm'] == 'minmax':
        # 1. Reverse normalization using min_coords and max_coords
        min_coords = data_batch.min_coords.to(device).view(-1, 2)
        max_coords = data_batch.max_coords.to(device).view(-1, 2)

        range_coords = max_coords - min_coords
        converted_output = (output + 1) / 2 * range_coords + min_coords


    elif params['norm'] == 'unit_sphere':
        # 1. Reverse unit sphere normalization using max_radius
        max_radius = data_batch.max_radius.to(device).view(-1, 1)
        converted_output = output * max_radius

    # 2. Reverse centering using the stored node_positions_center
    centers = data_batch.node_positions_center.to(device).view(-1, 2)
    converted_output += centers

    # return torch.tensor(converted_output, device=device)
    return converted_output.clone().detach().to(device)


def save_reduced_dataset(dataset, indices, path):
    """
    Saves only the necessary data from the original dataset at specified indices,
    effectively reducing the file size by excluding unnecessary data.
    """
    reduced_data = [dataset[i] for i in indices]  # Extract only the relevant data
    torch.save(reduced_data, path)  # Save the truly reduced dataset


def split_datasets(data):
    """
    Save the preprocessed data into train, validation, and test datasets.

    Args:
        preprocessed_data (pd.DataFrame): The preprocessed data to be split and saved.
        params (str): The GNN project parameters.
        experiments_path (str): The file path to save the train, test, validation datasets.

    Returns:
        Tuple[list, list, list, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        The train, validation, and test datasets and their corresponding DataFrames.
    """

    logging.info('Creating train test splits...')

    # Stratified split using scikit-learn
    train_idx, test_idx, train_test_y, test_y = train_test_split(
        np.arange(len(data)),
        data['dataset'],
        test_size=0.3,
        stratify=data['dataset'],
        random_state=100
    )

    # split the test into validation and test
    val_idx, test_idx, _, _ = train_test_split(
        test_idx,
        test_y,
        test_size=len(data) - len(train_idx) - int(0.1 * len(data)),
        stratify=test_y,
        random_state=100
    )

    # Convert indices back to DataFrame subsets
    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)

    # return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, raw_test_df
    return train_df, val_df, test_df


def save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path):
    """
    Process the combined train, validation, and test data, and save them to disk as .pkl files.

    Args:
        combined_train_df (pd.DataFrame): DataFrame containing combined training data.
        combined_val_df (pd.DataFrame): DataFrame containing combined validation data.
        combined_test_df (pd.DataFrame): DataFrame containing combined test data.
        experiments_path (str): The path where the processed data will be saved.
    """
    logging.info("Saving data")

    # Define file paths
    train_file_path = os.path.join(experiments_path, 'train_dataset.pkl')
    val_file_path = os.path.join(experiments_path, 'val_dataset.pkl')
    test_file_path = os.path.join(experiments_path, 'test_dataset.pkl')

    # Save the combined DataFrame subsets as .pkl files
    with open(train_file_path, 'wb') as f:
        pickle.dump(combined_train_df, f)
    with open(val_file_path, 'wb') as f:
        pickle.dump(combined_val_df, f)
    with open(test_file_path, 'wb') as f:
        pickle.dump(combined_test_df, f)

    # Dataset types for specific filtering
    if params['dynamic']:
        dataset_types = ['guided_path_data', 'linear_path_data']
    else:
        dataset_types = ['circle', 'triangle', 'rectangle', 'random', 'circle_jammer_outside_region',
                         'triangle_jammer_outside_region', 'rectangle_jammer_outside_region',
                         'random_jammer_outside_region', 'all_jammed', 'all_jammed_jammer_outside_region']

    for dataset in dataset_types:
        # Create filtered subsets based on dataset type
        train_subset = combined_train_df[combined_train_df['dataset'] == dataset]
        val_subset = combined_val_df[combined_val_df['dataset'] == dataset]
        test_subset = combined_test_df[combined_test_df['dataset'] == dataset]

        # Save each subset as .pkl if it is not empty
        if not train_subset.empty:
            train_subset_path = os.path.join(experiments_path, f'{dataset}_train_set.pkl')
            with open(train_subset_path, 'wb') as f:
                pickle.dump(train_subset, f)

        if not val_subset.empty:
            val_subset_path = os.path.join(experiments_path, f'{dataset}_val_set.pkl')
            with open(val_subset_path, 'wb') as f:
                pickle.dump(val_subset, f)

        if not test_subset.empty:
            test_subset_path = os.path.join(experiments_path, f'{dataset}_test_set.pkl')
            with open(test_subset_path, 'wb') as f:
                pickle.dump(test_subset, f)

def downsample_data(instance):
    """
    Downsamples the data of an instance object based on a fixed number of maximum nodes.

    Args:
        instance (Instance): The instance to downsample.
    """
    max_nodes = params['max_nodes']
    num_original_nodes = len(instance.node_positions)

    if num_original_nodes <= max_nodes:
        return instance  # No downsampling needed

    window_size = num_original_nodes // max_nodes
    num_windows = max_nodes

    # Create downsampled attributes
    downsampled_positions = []
    downsampled_noise_values = []
    downsampled_angles = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        downsampled_positions.append(np.mean(instance.node_positions[start_idx:end_idx], axis=0))
        downsampled_noise_values.append(np.mean(instance.node_noise[start_idx:end_idx]))
        if 'angle_of_arrival' in params['required_features']:
            downsampled_angles.append(np.mean(instance.angle_of_arrival[start_idx:end_idx]))

    # Update instance with downsampled data
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise_values)
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = np.array(downsampled_angles)

    return instance

# Noise based downsampling
def downsample_data_by_highest_noise(instance):
    """
    Downsamples the data of an instance object by keeping the nodes with the highest noise values.

    Args:
        instance (Instance): The instance to downsample.
        num_nodes_to_keep (int): The number of nodes to retain based on the highest noise values.

    Returns:
        Instance: The downsampled instance.
    """
    # Get the indices of the nodes with the highest noise values
    top_indices = np.argsort(instance.node_noise)[-params['max_nodes']:]

    # Sort the indices to maintain the order in the dataset
    top_indices = np.sort(top_indices)

    # Extract the corresponding positions, noise values, and angles (if applicable)
    downsampled_positions = instance.node_positions[top_indices]
    downsampled_noise_values = instance.node_noise[top_indices]

    if 'angle_of_arrival' in params['required_features']:
        downsampled_angles = instance.angle_of_arrival[top_indices]
        instance.angle_of_arrival = np.array(downsampled_angles)

    # Update instance with the downsampled data
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise_values)

    return instance


def bin_nodes(nodes, grid_meters):
    """Bin nodes by averaging positions, noise levels, and angle of arrival within each grid cell."""
    max_nodes = params['max_nodes']
    nodes['x_bin'] = (nodes['x'] // grid_meters).astype(int)
    nodes['y_bin'] = (nodes['y'] // grid_meters).astype(int)
    binned = nodes.groupby(['x_bin', 'y_bin']).mean().reset_index()
    binned['x'] = (binned['x_bin'] + 0.5) * grid_meters
    binned['y'] = (binned['y_bin'] + 0.5) * grid_meters
    # Sort by noise_level and keep the top max_nodes
    binned = binned.sort_values(by='noise_level', ascending=False).head(max_nodes)
    return binned


def hybrid_downsampling_pipeline(instance):
    """
    Combines spatial binning, time window averaging, and noise filtering in sequence.

    Args:
        instance: The data instance to downsample.
        filtering_proportion: Proportion of nodes to retain after noise filtering (e.g., 0.6 for 60%).

    Returns:
        instance: Downsampled instance.
    """
    # Step 1: Spatial Binning
    node_df = pd.DataFrame({
        'x': instance.node_positions[:, 0],
        'y': instance.node_positions[:, 1],
        'noise_level': instance.node_noise
    })
    if 'angle_of_arrival' in params['required_features']:
        node_df['angle_of_arrival'] = instance.angle_of_arrival

    binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])

    # Step 2: Time Window Averaging
    downsampled_positions, downsampled_noise, downsampled_angles = [], [], []
    max_nodes = params['max_nodes']
    num_binned_nodes = len(binned_nodes)

    window_size = max(1, num_binned_nodes // max_nodes)
    for i in range(0, num_binned_nodes, window_size):
        batch = binned_nodes.iloc[i:i + window_size]
        downsampled_positions.append(batch[['x', 'y']].mean().to_numpy())
        downsampled_noise.append(batch['noise_level'].mean())
        if 'angle_of_arrival' in params['required_features']:
            downsampled_angles.append(batch['angle_of_arrival'].mean())

    # Update instance after time window averaging
    instance.node_positions = np.array(downsampled_positions)
    instance.node_noise = np.array(downsampled_noise)
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = np.array(downsampled_angles)

    # Step 3: Noise Filtering
    num_filtered_nodes = max(1, int(max_nodes * params['filtering_proportion']))
    high_noise_indices = np.argsort(instance.node_noise)[-num_filtered_nodes:]
    instance.node_positions = instance.node_positions[high_noise_indices]
    instance.node_noise = instance.node_noise[high_noise_indices]
    if 'angle_of_arrival' in params['required_features']:
        instance.angle_of_arrival = instance.angle_of_arrival[high_noise_indices]

    return instance


def add_jammed_column(data, threshold=-55):
    data['jammed_at'] = None
    for i, noise_list in enumerate(data['node_noise']):
        # print("noise list: ", noise_list)
        # Check if noise_list is a valid non-empty list
        if not isinstance(noise_list, list) or len(noise_list) == 0:
            raise ValueError(f"Invalid or empty node_noise list at row {i}")

        count = 0
        jammed_index = None  # Store the index of the third noise > threshold

        for idx, noise in enumerate(noise_list):
            if noise > threshold:
                count += 1
                # Save the index of the third noise sample that exceeds the threshold
                if count == 3:
                    jammed_index = idx
                    break

        # Save the index of the third "jammed" sample or handle no jamming detected
        if jammed_index is not None:
            data.at[i, 'jammed_at'] = jammed_index
        else:
            raise ValueError(f"No sufficient jammed noise samples found for row {i}")

    return data


def load_data(params, test_set_name, experiments_path=None):
    """
    Load the data from the given paths, or preprocess and save it if not already done.

    Args:
        dataset_path (str): The file path of the raw dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        The train, validation, and test datasets.
    """
    if params['inference']:
        # Load the test data only for inference mode
        test_set_name = [test_set_name]
        for test_data in test_set_name:
            print(f"dataset: {test_data}")
            file_path = os.path.join(experiments_path, f'{test_data}.pkl')
            with open(file_path, 'rb') as f:
                test_df = pickle.load(f)
            print(test_df.columns)
            return None, None, test_df
    else:
        # Define file paths for train, validation, and test datasets
        train_file = os.path.join(experiments_path, 'train_dataset.pkl')
        val_file = os.path.join(experiments_path, 'val_dataset.pkl')
        test_file = os.path.join(experiments_path, 'test_dataset.pkl')

        if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            # Load existing datasets if they already exist
            logging.info("Loading train test data...")

            with open(train_file, 'rb') as f:
                train_df = pickle.load(f)
            with open(val_file, 'rb') as f:
                val_df = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_df = pickle.load(f)

        else:
            if params['all_env_data']:
                datasets = ['data/train_test_data/log_distance/urban_area/combined_urban_area.csv', 'data/train_test_data/log_distance/shadowed_urban_area/combined_shadowed_urban_area.csv']
            else:
                datasets = [params['dataset_path']]
            for dataset in datasets:
                print(f"dataset: {dataset}")

                # Load the interpolated dataset from the pickle file
                with open(dataset, "rb") as f:
                    data_list = []
                    try:
                        # Read each dictionary entry in the list and add to data_list
                        while True:
                            data_list.append(pickle.load(f))
                    except EOFError:
                        pass  # End of file reached

                # # Load the entire list from the pickle file in one go
                # data_list = []
                # with open(dataset, 'rb') as f:
                #     data_list = pickle.load(f)

                # Convert the list of dictionaries to a DataFrame
                data = pd.DataFrame(data_list)

                print(len(data))
                print("COLUMNS: ", data.columns)

                # Add additional columns required for processing
                data['id'] = range(1, len(data) + 1)
                data['dataset'] = 'linear_data'

                # data = pd.read_csv(dataset)
                # data['id'] = range(1, len(data) + 1)
                # convert_data_type(data, load_saved_data=False)

                # Add jammed column
                data = add_jammed_column(data, threshold=-55)

                # Create train test splits
                train_df, val_df, test_df = split_datasets(data)

            # Process and save the combined data
            save_datasets(train_df, val_df, test_df, experiments_path)

        return train_df, val_df, test_df


def get_params_hash(params):
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def create_data_loader(params, train_data, val_data, test_data, experiment_path):
    """
    Create data loaders using the TemporalGraphDataset instances for training, validation, and testing sets.
    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        val_data (pd.DataFrame): DataFrame containing the validation data.
        test_data (pd.DataFrame): DataFrame containing the testing data.
        batch_size (int): The size of batches.
    Returns:
        tuple: Three DataLoaders for the training, validation, and testing datasets.
    """
    if params['inference']:
        logging.info('Computing testing data')
        test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=params['test_discrite_coeff'])
        test_loader = DataLoader(test_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, num_workers=0)

        return None, None, test_loader
    else:
        # Generate a unique identifier for the current params
        params_hash = get_params_hash(params)
        cache_path = os.path.join(experiment_path, f"data_loader_{params_hash}.pkl")
        os.makedirs(experiment_path, exist_ok=True)

        if os.path.exists(cache_path):
            # Load cached data loaders
            with open(cache_path, 'rb') as f:
                train_loader, val_loader, test_loader = pickle.load(f)
            logging.info("Loaded cached data loaders")
        else:
            # Create data loaders and save them if cache doesn't exist
            logging.info("Creating data loaders")
            train_loader, val_loader, test_loader = generate_data_loaders(params, train_data, val_data, test_data)

            # Save data loaders to cache
            with open(cache_path, 'wb') as f:
                pickle.dump((train_loader, val_loader, test_loader), f)
            logging.info("Saved data loaders")

        return train_loader, val_loader, test_loader


def generate_data_loaders(params, train_data, val_data, test_data):
    train_dataset = TemporalGraphDataset(train_data, test=False)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=params['num_workers'])

    val_dataset = TemporalGraphDataset(val_data, test=True, discretization_coeff=params['val_discrite_coeff'])
    val_loader = DataLoader(val_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=params['test_discrite_coeff'])
    test_loader = DataLoader(test_dataset, batch_size=params['test_batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader, test_loader


def safe_convert_list(row: str, data_type: str):
    """
    Safely convert a string representation of a list to an actual list,
    with type conversion tailored to specific data types including handling
    for 'states' which are extracted and stripped of surrounding quotes.

    Args:
        row (str): String representation of a list.
        data_type (str): The type of data to convert ('jammer_pos', 'drones_pos', 'node_noise', 'states').

    Returns:
        List: Converted list or an empty list if conversion fails.
    """
    try:
        if data_type == 'jammer_position':
            result = row.strip('[').strip(']').split(', ')
            return [float(pos) for pos in result]
        elif data_type == 'node_positions':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        elif data_type == 'node_noise':
            result = row.strip('[').strip(']').split(', ')
            return [float(noise) for noise in result]
        elif data_type == 'node_rssi':
            result = row.strip('[').strip(']').split(', ')
            return [float(rssi) for rssi in result]
        elif data_type == 'node_states':
            result = row.strip('[').strip(']').split(', ')
            return [int(state) for state in result]
        elif data_type == 'timestamps':
            result = row.strip('[').strip(']').split(', ')
            return [float(time) for time in result]
        elif data_type == 'angle_of_arrival':
            result = row.strip('[').strip(']').split(', ')
            return [float(aoa) for aoa in result]
        elif data_type == 'jammed_at':
            return int(row)
        elif data_type == 'jammer_power':
            return float(row)
        elif data_type == 'num_samples':
            return float(row)
        elif data_type == 'sigma':
            return float(row)
        elif data_type == 'jammer_power':
            return float(row)
        elif data_type == 'id':
            return int(row)
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error

# Plot without edges and annotations
def plot_graph(positions, edge_index, node_features, edge_weights=None, jammer_positions=None, show_weights=False, perc_completion=None, id=None, jammer_power=None):
    G = nx.Graph()

    # Ensure positions and features are numpy arrays for easier handling
    positions = np.array(positions)
    node_features = np.array(node_features)
    if jammer_positions is not None:
        jammer_positions = np.array(jammer_positions)

    # Add nodes with features and positions
    for i, pos in enumerate(positions):
        # assuming RSSI is the last feature in node_features array
        if params['dynamic']:
            G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2],
                       timestamp=node_features[i][-1], sin_aoa=node_features[i][-3], cos_aoa=node_features[i][-2])
        else:
            G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2])

    # Position for drawing
    pos = {i: (p[0], p[1]) for i, p in enumerate(positions)}

    # Distinguish nodes based on noise value > -55
    noise_values = np.array([G.nodes[i]['noise'] for i in G.nodes()])
    node_colors = ['red' if noise > -55 else 'blue' for noise in noise_values]

    # Draw the graph nodes without edges and annotations
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1)

    # Optionally draw jammer position without annotations
    # Optionally draw jammer position without annotations
    if jammer_positions is not None:
        for i, jammer_pos in enumerate(jammer_positions):
            plt.scatter(*jammer_pos, color='red', s=100, marker='x', label='Jammer')  # Add jammer to the plot as a cross

            # Assuming jammer_power is available in an array, where each entry corresponds to a jammer
            # Add annotation for jammer_power
            plt.annotate(f'Power: {jammer_power:.1f} dB',
                         xy=jammer_pos,
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=10, color='black')

    perc_completion_title = "Graph " + str(id) + " " + str(round(perc_completion, 2)) + "% trajectory since start"
    plt.title(perc_completion_title, fontsize=15)
    plt.axis('off')  # Turn off the axis
    plt.show()

# ORIGINAL
def create_torch_geo_data(instance: Instance) -> Data:
    """
    Create a PyTorch Geometric Data object from a row of the dataset.

    Args:
        row (pd.Series): A row of the dataset containing drone positions, states, RSSI values, and other features.

    Returns:
        Data: A PyTorch Geometric Data object containing node features, edge indices, edge weights, and target variables.
    """
    # Downsample (binning and highest noise)
    if params['ds_method'] == 'noise':
        # Convert positions to a DataFrame to use bin_nodes
        if 'angle_of_arrival' in params['required_features']:
            node_df = pd.DataFrame({
                'x': instance.node_positions[:, 0],
                'y': instance.node_positions[:, 1],
                'noise_level': instance.node_noise,
                'angle_of_arrival': instance.angle_of_arrival  # Include angle of arrival
            })
            binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])
            instance.node_positions = binned_nodes[['x', 'y']].to_numpy()
            instance.node_noise = binned_nodes['noise_level'].to_numpy()
            instance.angle_of_arrival = binned_nodes['angle_of_arrival'].to_numpy()  # Update angle of arrival
        else:
            # Convert positions to a DataFrame to use bin_nodes
            node_df = pd.DataFrame({
                'x': instance.node_positions[:, 0],
                'y': instance.node_positions[:, 1],
                'noise_level': instance.node_noise
            })
            binned_nodes = bin_nodes(node_df, grid_meters=params['grid_meters'])
            instance.node_positions = binned_nodes[['x', 'y']].to_numpy()
            instance.node_noise = binned_nodes['noise_level'].to_numpy()

        # instance = downsample_data_by_highest_noise(instance)
    elif params['ds_method'] == 'time_window_avg':
        instance = downsample_data(instance)
    elif params['ds_method'] == 'hybrid':
        instance = hybrid_downsampling_pipeline(instance)
    else:
        raise ValueError("Undefined downsampling method")

    # Preprocess instance data
    center_coordinates_instance(instance)
    if params['norm'] == 'minmax':
        apply_min_max_normalization_instance(instance)
    elif params['norm'] == 'unit_sphere':
        apply_unit_sphere_normalization(instance)

    if 'angle_of_arrival' in params['required_features']:
        # Convert AoA from degrees to radians
        aoa_radians = np.radians(instance.angle_of_arrival)

        # Create node features without adding an extra list around the numpy array
        node_features = np.concatenate([
            instance.node_positions,
            instance.node_noise[:, None],  # Ensure node_noise is reshaped to (n, 1)
            np.sin(aoa_radians[:, None]),
            np.cos(aoa_radians[:, None])
        ], axis=1)
    else:
        node_features = np.concatenate([
            instance.node_positions,
            instance.node_noise[:, None]
        ], axis=1)


    # Convert to 2D tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    # Preparing edges and weights
    positions = instance.node_positions
    if params['num_neighbors'] == 'fc':
        num_neighbors = 10000000
    else:
        num_neighbors = params['num_neighbors']
    if params['edges'] == 'knn':
        num_samples = positions.shape[0]
        k = min(num_neighbors, num_samples - 1)  # num of neighbors, ensuring k < num_samples
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        edge_index, edge_weight = [], []

        # Add self loop
        for i in range(indices.shape[0]):
            edge_index.extend([[i, i]])
            edge_weight.extend([0.0])
            for j in range(1, indices.shape[1]):
                edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
                edge_weight.extend([distances[i, j], distances[i, j]])
    else:
        raise ValueError("Unsupported edge specification")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # jammer_positions = np.array(instance.jammer_position).reshape(-1, 2)  # Assuming this reshaping is valid based on your data structure

    if not params['inference']:
        # Assuming instance.jammer_position is a list or array that can be reshaped
        jammer_positions = np.array(instance.jammer_position).reshape(-1, 2)

        # Convert jammer_positions to a tensor
        y = torch.tensor(jammer_positions, dtype=torch.float)

        # Convert instance.jammer_power to a tensor and reshape it to match the dimensions of y
        jammer_power = torch.tensor(instance.jammer_power, dtype=torch.float).reshape(-1, 1)

        # Concatenate jammer_power to y along the appropriate dimension
        y = torch.cat((y, jammer_power), dim=1)
    else:
        y = None

    # Plot
    # plot_graph(positions=positions, edge_index=edge_index, node_features=node_features_tensor, edge_weights=edge_weight, jammer_positions=jammer_positions, show_weights=True, perc_completion=instance.perc_completion, id=instance.id, jammer_power=instance.jammer_power)

    # Create the Data object
    data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # Convert geometric information to tensors
    # data.id = instance.id
    data.node_positions_center = torch.tensor(instance.node_positions_center, dtype=torch.float)
    if params['norm'] == 'minmax':
        data.min_coords = torch.tensor(instance.min_coords, dtype=torch.float)
        data.max_coords = torch.tensor(instance.max_coords, dtype=torch.float)
    elif params['norm'] == 'unit_sphere':
        data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)

    # Store the perc_completion as part of the Data object
    if params['inference']:
        instance.perc_completion = 1
    data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)

    return data
