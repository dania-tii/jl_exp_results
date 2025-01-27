params = {
    'model': 'GCN',
    'learning_rate': 0.0004815321758405571,
    'weight_decay': 0.00005,
    'test_batch_size': 32,
    'batch_size': 32,
    'dropout_rate': 0.5,
    'num_heads': 8,
    'num_layers': 2,
    'hidden_channels': 512,
    'out_channels': 256,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'max_epochs': 200,
    '3d': False,
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['mean_noise', 'std_noise', 'range_noise', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth'],
    'num_neighbors': 10,
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'unit_sphere',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'max_nodes': 400,
    'filtering_proportion': 0.6,
    'grid_meters': 1,
    'ds_method': 'hybrid', # time_window_avg, noise
    'experiments_folder': 'dynamic/interpolat/',
    'dataset_path': 'data/modified_interpolated_controlled_path_data_1000.pkl',  # combined_fspl_log_distance.csv',  # combined_urban_area.csv
    'test_sets': ['guided_path_data_test_set.csv', 'linear_path_data_test_set.csv'],
    'dynamic': True,
    'train_per_class': False,
    'all_env_data': False,
    'inference': False,
    'reproduce': True,
    'plot_network': False,
    'study': 'downsampling',  # dataset, coord_system, feat_engineering, knn_edges, downsampling
    'val_discrite_coeff': 0.25,
    'test_discrite_coeff': 0.1,  # disable discritization -> step_size = 1
    'num_workers': 16,
    'aug': ['crop']
}

