import torch
import torch_geometric as ptg
import numpy as np
import networkx as nx
import json
import os
import time
import pickle
import argparse
import wandb
import copy
from tqdm import tqdm

from lib.cg_batch import cg_batch
import visualization as vis
import vi
from dgmrf import DGMRF
import constants
import utils
import inference_new

from temporal_model import TemporalModel
import matplotlib.pyplot as plt

def get_config():
    parser = argparse.ArgumentParser(description='Graph DGMRF')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")


    # Arguments of importance
    parser.add_argument("--n_layers", type=int,
                    help="Number of message passing layers", default=2) 
    parser.add_argument("--n_layers_temporal", type=float, default=0,
        help="Number of layers in temporal model")
    parser.add_argument("--n_iterations", type=int,
        help="How many iterations to train for", default=1000) #1000
    parser.add_argument("--n_training_samples", type=int, default=10, #10
        help="Number of samples to use for each iteration in training")
    parser.add_argument("--sample_times_start", type=float, default=3,
        help="Start sample time")
    parser.add_argument("--sample_times_end", type=float, default=3,
        help="End sample time")
    parser.add_argument("--seed", type=int, default=1,
        help="Seed for random number generator")

    # General
    parser.add_argument("--dataset", type=str, default="advection_diffusion", 
            help="Which dataset to use")
    parser.add_argument("--noise_std", type=int, default=1e-2,
            help="Value to use for noise std.-dev. (if not learned, otherwise initial)")
    parser.add_argument("--learn_noise_std", type=int, default=1,
            help="If the noise std.-dev. should be learned jointly with the model")
    parser.add_argument("--optimizer", type=str, default="adam",
            help="Optimizer to use for training")
    parser.add_argument("--print_params", type=int, default=0,
            help="Write out parameter values during training")
    parser.add_argument("--features", type=int, default=0,
            help="Include additional node-features, apart from Gaussian field")
    parser.add_argument("--coeff_inv_std", type=float, default=0.0001,
            help="Inverse standard deviation of coefficients beta (feature weights)")

    # Model Architecture
    parser.add_argument("--use_bias", type=int, default=1,
                        help="Use bias parameter in layers")
    parser.add_argument("--non_linear", type=int, default=0,
                        help="Add in non-linear layers, requiring VI for posterior")
    parser.add_argument("--dist_weight", type=int, default=0,
                        help="Use distance weighted adjacency matrix")
    parser.add_argument("--fix_gamma", type=int, default=0,
                        help="If the value of the gamma parameter should be fixed")
    parser.add_argument("--gamma_value", type=float, default=1.0,
                        help="Value for gamma when fixed")

    # Training
    parser.add_argument("--log_det_method", type=str, default="eigvals",
        help="Method for log-det. computations (eigvals/dad), dad is using power series")
    parser.add_argument("--lr", type=float,
            help="Learning rate", default=0.01)
    parser.add_argument("--val_interval", type=int, default=10**2, #100
            help="Evaluate model every val_interval:th iteration")
    parser.add_argument("--vi_layers", type=int, default=1,
        help="Flex-layers to apply to independent vi-samples to introduce correlation")

    # Posterior inference
    parser.add_argument("--n_post_samples", type=int, default=100,
        help="Number of samples to draw from posterior for MC-estimate of std.-dev.")
    parser.add_argument("--vi_eval", type=int, default=0,
        help="Use variational distribution in place of true posterior, in evaluation")
    parser.add_argument("--inference_rtol", type=float, default=1e-7,
            help="rtol for CG during inference")

    # Plotting
    parser.add_argument("--plot_vi_samples", type=int, default=3,
        help="Number of vi samples to plot")
    parser.add_argument("--plot_post_samples", type=int, default=3,
        help="Number of posterior samples to plot")
    parser.add_argument("--save_pdf", type=int, default=1,
        help="If plots should also be saved as .pdf-files")
    parser.add_argument("--dump_prediction", type=int, default=0,
        help="If produced graphs should be saved to files")
    
    # N args
    parser.add_argument("--n_lattice", type=int, default=30,
        help="Number of lattice points in each dimension in the spatial graph")
    parser.add_argument("--n_space", type=int, default=30**2,
        help="Number of spatial points at each time step")
    parser.add_argument("--n_time", type=float, default=20,
        help="Number of time steps")
    

    args = parser.parse_args()
    config = vars(args)

    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        with open(args.config) as json_file:
            config_from_file = json.load(json_file)

        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    return config

## Get config parameters

config = get_config()


## Load data
    
dataset_dict = utils.load_dataset(config["dataset"])


## Time points
    
if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
    config["n_time"] = config["sample_times_end"] - config["sample_times_start"] + 1


## Create models

# Instatiate temporal model 
temporal_model = TemporalModel(config)

# Instatiate spatial model and variational distribution 
# Spatial model: an independent DGMRF for each time step, cf. section 3.2.1
# Variational distribution: an independent variational distribution for each time step, cf. section 3.3.1
dgmrf_list = []
vi_dist_list = []
for k in range(config["n_time"]):
    # Graph data
    if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
        graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
    else:
        graph_k = dataset_dict["graph_y_" + str(k)]
    
    # Instantiate spatial model 
    dgmrf_k = DGMRF(graph_k, config)

    # Instantiate vi_dist for each time step
    vi_dist_k = vi.VariationalDist(config, graph_k)

    # Append to list
    dgmrf_list.append(dgmrf_k)
    vi_dist_list.append(vi_dist_k)


## Load optimal parameters

best_temporal_params = torch.load('results/best_temporal_params.pt')
best_spatial_params = torch.load('results/best_spatial_params.pt')
best_vi_params = torch.load('results/best_vi_params.pt')

temporal_model.load_state_dict(best_temporal_params)
for k in range(config["n_time"]):
    dgmrf_list[k].load_state_dict(best_spatial_params[k])
    vi_dist_list[k].load_state_dict(best_vi_params[k])

# Add noise std to opt_param if learn_noise_std == 1
config["noise_std"] = torch.load('results/noise_std.pt')
config["log_noise_std"] = torch.log(config["noise_std"])


## Load post mean true and post mean model
    
post_mean_true = dataset_dict["post_mean_true"]
post_mean_model = torch.load('results/post_mean_model_2_1_1000_10_2_3.pt')




## Plot obs and predicted obs

n_time = config["n_time"]
n_lattice = config["n_lattice"] 

# Create a figure to hold all subplots
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

# Loop through each time step
for k in range(n_time):
    # Graph data
    if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
        graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
    else:
        graph_k = dataset_dict["graph_y_" + str(k)]

    # Obs
    obs_k = graph_k.x.squeeze(1)

    # Mask the data with the graph
    obs_mask_k = graph_k.mask * obs_k


    # Reshape the masked data for true mean at time step k for plotting
    obs_mask_k = obs_mask_k.reshape((n_lattice, n_lattice))
    # Reshape the data for true mean at time step k for plotting
    obs_k = obs_k.reshape((n_lattice, n_lattice))
    # Reshape the data for model mean at time step k for plotting
    post_mean_model_k = post_mean_model[:, k].reshape((n_lattice, n_lattice))
    
    # Plot for masked true mean at time step k
    plt.subplot(3, n_time, k + 1)  # Arguments are (rows, columns, subplot index)
    plt.imshow(obs_mask_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"Obs masked: Time Step {k}")  # Add a title to each subplot

    # Plot for true mean at time step k
    plt.subplot(3, n_time, n_time + k + 1)  # Arguments are (rows, columns, subplot index)
    plt.imshow(obs_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"Obs: Time Step {k}")  # Add a title to each subplot
    
    # Plot for model mean at time step k
    plt.subplot(3, n_time, 2 * n_time + k + 1)  # Adjust the index for the bottom row
    plt.imshow(post_mean_model_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"Model Mean: Time Step {k}")  # Add a title to each subplot

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.show()





## Plot post mean true and post mean model


# Create a figure to hold all subplots
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

# Loop through each time step
for k in range(n_time):
    # Graph data
    if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
        graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
    else:
        graph_k = dataset_dict["graph_y_" + str(k)]

    # Mask the data with the graph
    post_mean_true_mask_k = graph_k.mask * post_mean_true[:, k]

    # Reshape the masked data for true mean at time step k for plotting
    post_mean_true_mask_k = post_mean_true_mask_k.reshape((n_lattice, n_lattice))
    # Reshape the data for true mean at time step k for plotting
    post_mean_true_k = post_mean_true[:, k].reshape((n_lattice, n_lattice))
    # Reshape the data for model mean at time step k for plotting
    post_mean_model_k = post_mean_model[:, k].reshape((n_lattice, n_lattice))
    
    # Plot for masked true mean at time step k
    plt.subplot(3, n_time, k + 1)  # Arguments are (rows, columns, subplot index)
    plt.imshow(post_mean_true_mask_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"True Masked Mean: Time Step {k}")  # Add a title to each subplot

    # Plot for true mean at time step k
    plt.subplot(3, n_time, n_time + k + 1)  # Arguments are (rows, columns, subplot index)
    plt.imshow(post_mean_true_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"True Mean: Time Step {k}")  # Add a title to each subplot
    
    # Plot for model mean at time step k
    plt.subplot(3, n_time, 2 * n_time + k + 1)  # Adjust the index for the bottom row
    plt.imshow(post_mean_model_k)  # Replace with the correct data
    plt.colorbar()
    plt.title(f"Model Mean: Time Step {k}")  # Add a title to each subplot

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.show()


import numpy as np
x = [0.531997, 0.9337842, 0.6823764, 0.7558424, 0.7878496]
np.mean(x)