import torch
import numpy as np
import json
import os
import time
import argparse
import copy
from temporal_model import TemporalModel
from dgmrf import DGMRF
import vi
import utils
import inference



def get_config():
    parser = argparse.ArgumentParser(description='Graph DGMRF')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # Arguments of importance
    parser.add_argument("--n_layers", type=int,
                    help="Number of message passing layers", default=2) 
    parser.add_argument("--n_layers_temporal", type=float, default=3,
        help="Number of layers in temporal model")
    parser.add_argument("--n_iterations", type=int,
        help="How many iterations to train for", default=1000) #1000
    parser.add_argument("--n_training_samples", type=int, default=100, #10
        help="Number of samples to use for each iteration in training")
    parser.add_argument("--val_interval", type=int, default=10, 
        help="Evaluate model every val_interval:th iteration")
    parser.add_argument("--sample_times_start", type=float, default=0,
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

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():

    # Get config parameters
    config = get_config()

    # Set all random seeds
    seed_all(config["seed"])

    # Device setup
    if torch.cuda.is_available():
        # Make all tensors created go to GPU
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_dict = utils.load_dataset(config["dataset"])
    for key in dataset_dict.keys():
      dataset_dict[key] = dataset_dict[key].to(device)
    
    # Initialize spatial graph using the first time step - all time steps have the same spatial graph 
    # graph_y = dataset_dict["graph_y_0"]

    # Time points
    if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
        config["n_time"] = config["sample_times_end"] - config["sample_times_start"] + 1

    # Initialize optimal parameters
    opt_params = ()

    # Instatiate temporal model 
    temporal_model = TemporalModel(config)
    opt_params += tuple(temporal_model.parameters())
    temporal_model.to(device)
    
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
        opt_params += tuple(dgmrf_k.parameters())
        dgmrf_k.to(device)

        # Instantiate vi_dist for each time step
        vi_dist_k = vi.VariationalDist(config, graph_k)
        opt_params += tuple(vi_dist_k.parameters())
        vi_dist_k.to(device)

        # Append to list
        dgmrf_list.append(dgmrf_k)
        vi_dist_list.append(vi_dist_k)

    # Add noise std to opt_param if learn_noise_std == 1
    config["log_noise_std"] = torch.log(torch.tensor(config["noise_std"]))
    if config["learn_noise_std"]:
        # Initalize using noise_std in config
        config["log_noise_std"] = torch.nn.parameter.Parameter(config["log_noise_std"])
        opt_params += (config["log_noise_std"],)

    # Set optimizer for stochastic gradient descent algorithm
    optimizer = utils.get_optimizer(config["optimizer"])(opt_params, lr=config["lr"])
    
    # Initialize loss variables
    total_loss = torch.zeros(1)
    elbo = torch.zeros(1)
    loss = torch.zeros(1)

    # Training loop
    best_loss = None
    best_temporal_params = None
    best_spatial_params = None

    # Start timing
    start_time = time.time()  

    for iteration_i in range(config["n_iterations"]):

        optimizer.zero_grad()
        elbo = torch.zeros(1)

        for k in range(config["n_time"]):

            ## Data

            # Graph data
            if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
            else:
                graph_k = dataset_dict["graph_y_" + str(k)]

            graph_k.n_observed = torch.sum(graph_k.mask).to(torch.float32)

            ## Train using variational distribution - i.e compute ELBO and use as loss function

            # Sample from variational distribution
            vi_dist_k = vi_dist_list[k]
            vi_samples = vi_dist_k.sample()

            # Prepare vi samples for batched temporal model
            # Hence, reshape to (n_samples, n_nodes, 1) for torch.bmm in temporal.py to work correctly
            vi_samples_temporal_batch_format = vi_samples.unsqueeze(2)

            # Feed samples through temporal model
            h_k = temporal_model(vi_samples_temporal_batch_format)

            # Prepare vi samples after temporal transform for batched spatial model
            vi_dist_k.sample_batch.x = h_k.reshape(-1,1)

            # Feed samples through spatial model - SHOULD WE HAVE A DGMRF FOR EACH TIME STEP - dgmrf_k, see section 3.2.1??
            dgmrf_k = dgmrf_list[k]
            g_k = dgmrf_k(vi_dist_k.sample_batch)

            # Compute log determinant of variational distribution
            vi_log_det = vi_dist_k.log_det()

            # Compute ELBO components for time step k
            l1 = 0.5 * vi_log_det
            l2 = -graph_k.n_observed * config["log_noise_std"]
            l3 = dgmrf_k.log_det()
            l4 = -(1./(2. * config["n_training_samples"])) * torch.sum(torch.pow(g_k,2))
            l5 = -(1./(2. * utils.noise_var(config)*\
                config["n_training_samples"])) * torch.sum(torch.pow(
                    (vi_samples - graph_k.x.flatten()), 2)[:, graph_k.mask])
            
            # Update ELBO
            elbo += l1 + l2 + l3 + l4 + l5

        # Compute normalized loss for this iteration
        n_nodes = config["n_space"] * config["n_time"]
        loss = (-1. / n_nodes) * elbo
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update total loss for reporting
        total_loss += loss.detach()

        # Print progress
        if True:#((iteration_i+1) % config["val_interval"]) == 0:
            # Initialize validation error accumulator
            val_error_accum = 0.0
            # Loop over time steps for validation
            for k in range(config["n_time"]):
                # Fetch the graph data for time step k
                if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                    graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
                else:
                    graph_k = dataset_dict["graph_y_" + str(k)]

                # Masked data for validation
                val_mask = torch.logical_not(graph_k.mask)
                graph_k.n_unobserved = torch.sum(val_mask).to(torch.float32)

                # Sample from the variational distribution
                vi_dist_k = vi_dist_list[k]
                val_samples = vi_dist_k.sample()

                # Calculate validation error for current time step
                if not graph_k.n_unobserved == 0:
                    val_error = (1./(config["n_training_samples"]*graph_k.n_unobserved)) *\
                                torch.sum(torch.pow((val_samples - graph_k.x.flatten()), 2)[:, val_mask])
                    val_error_accum += val_error.item()

            # Average validation error over time steps
            mean_val_error = val_error_accum / config["n_time"]

            # Compute mean loss over interval
            mean_loss = (total_loss.item() / config["val_interval"])
            total_loss = torch.zeros(1)

            print("Iteration: {}, mean loss: {:.6}, mean val error: {:.6}".format(
                (iteration_i+1), mean_loss, mean_val_error))

            # Save best parameters based on validation error
            if (best_loss is None) or (mean_loss < best_loss):
                best_temporal_params = copy.deepcopy(temporal_model.state_dict())   
                best_spatial_params = []
                best_vi_params = []
                for k in range(config["n_time"]):
                    best_spatial_params.append(copy.deepcopy(dgmrf_list[k].state_dict()))
                    best_vi_params.append(copy.deepcopy(vi_dist_list[k].state_dict()))
                best_loss = mean_loss

            if config["print_params"]:
                # Print parameters of both temporal and spatial models
                utils.print_params(temporal_model, config, model_type="temporal", header="--- Temporal Model parameters ---")
                for k in range(config["n_time"]):
                    utils.print_params(dgmrf_list[k], config, model_type="spatial", header="--- Spatial Model parameters (time step {}) ---".format(k))


    # # Summary
    # print("n_spatial_layers: ", config["n_layers"])
    # print("n_temporal_layers: ", config["n_layers_temporal"])
    # print("n_iterations: ", config["n_iterations"])
    # print("n_training_samples: ", config["n_training_samples"])
    # print("Iteration: {}, mean loss: {:.6}, mean val error: {:.6}".format(
    #     (iteration_i+1), mean_loss, mean_val_error))
    
    # End timing
    current_time = time.time()
    elapsed_time = (current_time - start_time) / 60
    print(f"Computation time for Stochastic Gradient Descent method: {elapsed_time:.2f} minutes")

    # Reload best parameters
    temporal_model.load_state_dict(best_temporal_params)
    for k in range(config["n_time"]):
        dgmrf_list[k].load_state_dict(best_spatial_params[k])
        vi_dist_list[k].load_state_dict(best_vi_params[k])
    
    # # Print final parameters 
    # utils.print_params(dgmrf, config, model_type="spatial", header="Final Spatial Model Parameters:")
    # utils.print_params(temporal_model, config, model_type="temporal", header="Final Temporal Model Parameters:")
    # if config["learn_noise_std"]:
    #     print("noise_std: {}".format(utils.noise_std(config)))

    # # Plot y
    # vis.plot_graph(graph_y, name="y", title="y")

    # These posteriors are over y
    vi_evaluation = config["vi_eval"] or config["non_linear"]
    if vi_evaluation:
        # Use variational distribution in place of true posterior
        print("Using variational distribution as posterior estimate ...")
        # graph_post_mean, graph_post_std = vi_dist.posterior_estimate(graph_y, config)
    else:
        # Exact posterior inference
        print("Running posterior inference ...")
        # graph_post_mean, graph_post_std = inference.posterior_inference(dgmrf,
        #         graph_y, config)

        # Start timing
        start_time = time.time() 

        post_mean_model = inference.posterior_inference(
            temporal_model=temporal_model, 
            dgmrf_list=dgmrf_list, 
            vi_dist_list=vi_dist_list,
            config=config,  
            dataset_dict=dataset_dict
        )
        
        # End timing
        current_time = time.time()
        elapsed_time = (current_time - start_time) / 60
        print(f"Computation time for posterior inference CG method: {elapsed_time:.2f} minutes")

    # NEXT UP: 
    # (1) Compute true posterior mean and std.-dev. for comparison - perhaps best to do so in advection_diffusion.py to include it in the dataset_dict
    # (2) Compute metrics for evaluation - RMSE, CRPS, INT
    

    # Compute Metrics
    if ("post_mean_true" in dataset_dict): #and ("graph_post_true_std" in dataset_dict):
        
        # Get true posterior mean from dataset_dict
        post_mean_true = dataset_dict["post_mean_true"]

        # Reshape post_mean_model to match shape of post_mean_true. Hence, from (n_time x n_nodes, 1) to (n_nodes, n_time) 
        post_mean_model = post_mean_model.reshape(config["n_time"], config["n_space"]).transpose(0,1)

        # Save post_mean_model to .pt file
        torch.save(post_mean_model, 
                   "results/post_mean_model_" + 
                   str(config['n_layers']) + "_" + 
                   str(config['n_layers_temporal']) + "_" + 
                   str(config['n_iterations']) + "_" + 
                   str(config['n_training_samples']) + "_" + 
                   str(config['sample_times_start']) + "_" + 
                   str(config['sample_times_end']) + 
                   ".pt")

        # Loop over time and compute metrics
        mae_total = 0.0
        rmse_total = 0.0
        for k in range(config["n_time"]):
            # Load graph
            if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
            else:
                graph_k = dataset_dict["graph_y_" + str(k)]

            # Compute metrics for unobserved nodes
            inverse_mask = torch.logical_not(graph_k.mask)
            diff = (post_mean_model[:,k] - post_mean_true[:,k])[inverse_mask]
            mae = torch.mean(torch.abs(diff))
            rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
            mae_total += mae
            rmse_total += rmse

            # Print metrics
            print("Time step: {}, MAE: {:.7}, RMSE: {:.7}".format(k, mae, rmse))

        print("Total MAE of posterior mean: {:.7}".format(mae_total))
        print("Total RMSE of posterior meam: {:.7}".format(rmse_total))

    
    # # Compare posterior mean with y
    # diff = (graph_post_mean.x - graph_y.x)[inverse_mask, :]
    # mae = torch.mean(torch.abs(diff))
    # rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))

    # pred_mean_np = graph_post_mean.x[inverse_mask, :].cpu().numpy()
    # pred_std_np = graph_post_std.x[inverse_mask, :].cpu().numpy()
    # target_np = graph_y.x[inverse_mask, :].cpu().numpy()

    # crps =  utils.crps_score(pred_mean_np, pred_std_np, target_np)
    # int_score = utils.int_score(pred_mean_np, pred_std_np, target_np)

    # print("MAE:  \t{:.7}".format(mae))
    # print("RMSE: \t{:.7}".format(rmse))
    # print("CRPS: \t{:.7}".format(crps))
    # print("INT:  \t{:.7}".format(int_score))
    # wandb.run.summary["mae"] = mae
    # wandb.run.summary["rmse"] = rmse
    # wandb.run.summary["crps"] = crps
    # wandb.run.summary["int_score"] = int_score

    # if "graph_x" in dataset_dict:
    #     # Plot x, if known
    #     graph_x = dataset_dict["graph_x"]
    #     vis.plot_graph(graph_x, name="x", title="x")

    # # Plot additional zooms for dataset
    # zoom_list = utils.get_dataset_zooms(config["dataset"])
    # for zoom_i, zoom in enumerate(zoom_list):
    #     # Plot y, posterior mean and posterior std for zooms
    #     vis.plot_graph(graph_y, name="y", title="y (zoom {})".format(zoom_i), zoom=zoom)
    #     vis.plot_graph(graph_post_mean, name="post_mean",
    #             title="Posterior Mean (zoom {})".format(zoom_i), zoom=zoom)
    #     vis.plot_graph(graph_post_std, name="post_std",
    #         title="Posterior Marginal Std.-Dev. (zoom {})".format(zoom_i), zoom=zoom)

    # # Optionally save prediction graphs
    # if config["dump_prediction"]:
    #     save_graphs = {"post_mean": graph_post_mean, "post_std": graph_post_std}
    #     if not vi_evaluation:
    #         save_graphs.update({"vi_mean": graph_vi_mean, "vi_std": graph_vi_std})

    #     for name, graph in save_graphs.items():
    #         utils.save_graph(graph, "{}_graph".format(name), wandb.run.dir)

if __name__ == "__main__":
    main()

