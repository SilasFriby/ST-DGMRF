import torch
import torch_geometric as ptg

from lib.cg_batch import cg_batch
import utils
import visualization as vis

def get_bias(dgmrf, graph_y):
    zero_graph = utils.new_graph(graph_y, new_x=torch.zeros(graph_y.num_nodes, 1))
    bias = dgmrf(zero_graph)
    return bias

# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, temporal_model, dgmrf, graph_y, config, n_time, dataset_dict, rtol, verbose=False):
    # # Only create the graph_batch once, then we can just replace x
    # n_nodes = graph_y.num_nodes
    # x_dummy = torch.zeros(rhs.shape[0],n_nodes,1)
    # graph_list = [utils.new_graph(graph_y, new_x=x_part) for x_part in x_dummy]
    # graph_batch = ptg.data.Batch.from_data_list(graph_list)

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)

    # zeros_temporal_batch_format = torch.zeros(rhs.shape[0], n_nodes_spatial, 1) 
    n_nodes_spatial = graph_y.num_nodes
    n_nodes = n_nodes_spatial * n_time

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, n_nodes, 1)
        # Implicitly applies posterior precision matrix Q_tilde to a vector x
        # y = (F^T @ S^T @ S @ F + (1/sigma^2)*I_masked)x

        # NEXT UP!!!!
        # 1) Add with_bias arg to temopral_model
        # 2) Consider removing batch dimension and then just add it in the end
        # 3) Create F^T as a sparse matrix and use torch.sparse.mm(F_T_sparse, Q_F_x)

        # Feed x_k through temporal model to F_k @ x_k
        F_x_k = temporal_model(x, batch_run=False, n_samples=1, with_bias=False)

        # Prepare F_x_k for spatial model
        graph_F_x_k = utils.new_graph(graph_y, new_x=F_x_k.squeeze(0))

        # Feed graph_F_k_x through spatial model to obtain S_k @ F_k @ x_k
        S_F_x_k = dgmrf(graph_F_x_k, with_bias=False)

        # Feed S_F_x_k through spatial model with transpose=True to obtain S^T_k @ F_k @ S_k @ x_k = Q_k @ F_k @ x_k
        Q_F_x_k = dgmrf(S_F_x_k, transpose=True, with_bias=False)

        # Create Q_F_x for all time points - i.e repeat Q_F_x_k n_time times, since Q is a block-diagonal matrix, with Q_k in the diagonal
        Q_F_x = torch.cat([Q_F_x_k for _ in range(n_time)], dim=0)

        ## F^T - block upper diagonal matrix with I in the diagonal anf F_k^T in the upper diagonal

        # Initialize F_k as an identity matrix
        F_k = torch.eye(config['n_lattice'] ** 2)

        # Multiply the F matrices of each TemporalLayer to get the final F
        for layer in temporal_model.layers:
            # You need to create the M matrix and add the identity matrix
            M = layer.create_matrix_M()
            F_layer = torch.eye(M.size(0)) + M
            F_k = torch.matmul(F_layer, F_k)  # Update F by multiplying by the new F_layer
        
        # F_k Transpose
        F_k_T = torch.transpose(F_k, 1, 2)

        # Fill F^T
        F_T = torch.zeros(n_time * n_nodes_spatial, n_time * n_nodes_spatial)

        # Fill F_T - upper bi-diagonal matrix
        for k in range(n_time):
            # Diagonal blocks with identity matrices
            idx_start = k * n_nodes_spatial
            idx_end = idx_start + n_nodes_spatial
            F_T[idx_start:idx_end, idx_start:idx_end] = torch.eye(n_nodes_spatial)

            # Superdiagonal blocks with F_k_T
            if k < n_time - 1:
                F_T[idx_start:idx_end, idx_end:(idx_end + n_nodes_spatial)] = F_k_T

        # Add batch dimension to F_T
        F_T = F_T.unsqueeze(0)

        # y first term: (F^T @ S^T @ S @ F)x
        y_first_term = torch.bmm(F_T, Q_F_x)

        ## mask

        for k in range(n_time):
            if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                graph_y = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
            else:
                graph_y = dataset_dict["graph_y_" + str(k)]

            mask_k = graph_y.mask.to(torch.float32).unsqueeze(1)  

            # Concatenate masked_y_k
            if k == 0:
                mask = mask_k
            else:
                mask = torch.cat((mask, mask_k), dim=0)

        # y second term: (1/sigma^2)y_masked
        y_second_term = (1./utils.noise_var(config)) * mask

        res = y_first_term.view(-1,n_nodes,1) + y_second_term.view(1,n_nodes,1)*x
        # Shape (n_batch, n_nodes,1)
        return res

    if config["features"]:
        # Feature matrix with 0-rows for masked nodes
        masked_features = graph_y.features * graph_y.mask.to(torch.float64).unsqueeze(1)
        masked_features_cov = masked_features.transpose(0,1)@masked_features

        noise_precision = 1/utils.noise_var(config)

        def Q_tilde_batched_with_features(x):
            # x has shape (n_batch, n_nodes+n_features, 1)
            node_x = x[:,:n_nodes]
            coeff_x = x[:,n_nodes:]

            top_res1 = Q_tilde_batched(node_x)
            top_res2 = noise_precision*masked_features@coeff_x

            bot_res1 = noise_precision*masked_features.transpose(0,1)@node_x
            bot_res2 = noise_precision*masked_features_cov@coeff_x +\
                    (config["coeff_inv_std"]**2)*coeff_x

            res = torch.cat((
                top_res1 + top_res2,
                bot_res1 + bot_res2,
                ), dim=1)

            return res

        Q_tilde_func = Q_tilde_batched_with_features
    else:
        Q_tilde_func = Q_tilde_batched

    solution, cg_info = cg_batch(Q_tilde_func, rhs, rtol=rtol)

    if verbose:
        print("CG finished in {} iterations, solution optimal: {}".format(
            cg_info["niter"], cg_info["optimal"]))

    return solution.to(torch.float32)

@torch.no_grad()
def sample_posterior(n_samples, dgmrf, graph_y, config, rtol, verbose=False):
    # Construct RHS using Papandeous and Yuille method
    bias = get_bias(dgmrf, graph_y)
    std_gauss1 = torch.randn(n_samples, graph_y.num_nodes, 1) - bias # Bias offset
    std_gauss2 = torch.randn(n_samples, graph_y.num_nodes, 1)

    std_gauss1_graphs = ptg.data.Batch.from_data_list(
        [utils.new_graph(graph_y, new_x=sample) for sample in std_gauss1])
    rhs_sample1 = dgmrf(std_gauss1_graphs, transpose=True, with_bias=False)
    rhs_sample1 = rhs_sample1.reshape(-1, graph_y.num_nodes, 1)

    float_mask = graph_y.mask.to(torch.float32).unsqueeze(1)
    y_masked = (graph_y.x * float_mask).unsqueeze(0)
    gauss_masked = std_gauss2 * float_mask.unsqueeze(0)
    rhs_sample2 = (1./utils.noise_var(config))*y_masked +\
        (1./utils.noise_std(config))*gauss_masked

    rhs_sample = rhs_sample1 + rhs_sample2
    # Shape (n_samples, n_nodes, 1)

    if config["features"]:
        # Change rhs to also sample coefficients
        n_features = graph_y.features.shape[1]
        std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
        rhs_sample1_coeff = config["coeff_inv_std"]*std_gauss1_coeff

        rhs_sample2_coeff = graph_y.features.transpose(0,1)@rhs_sample2
        rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff

        rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff), dim=1)

    # Solve using Conjugate gradient
    samples = cg_solve(rhs_sample, dgmrf, graph_y, config,
            rtol=rtol, verbose=verbose)

    return samples

@torch.no_grad()
def posterior_inference(temporal_model, dgmrf, config, graph_y, n_time, dataset_dict):
    
    #### Posterior mean
    #### RHS consists of Omega @ mu + (1/sigma^2)*y_masked

    ## Compute Omega @ mu = F^T @ Q @ c = F^T @ S^T @ S @ c

    if (config['temporal_time_invariance']):

        ## S @ c - Recall that ST-DGMRF model bias is b_theta = -(S @ c), cf. section 3.1.2
        
        # Feed zeros through temporal model to get temporal bias b_f
        n_nodes_spatial = graph_y.num_nodes
        zeros_temporal_format = torch.zeros(1, n_nodes_spatial, 1) 
        b_f_k = temporal_model(zeros_temporal_format, batch_run=False, n_samples=1)

        # Prepare b_f_k for spatial model
        graph_b_f_k = utils.new_graph(graph_y, new_x=b_f_k.squeeze(0))

        # Feed graph_b_f_k through spatial model to obtain model bias at time k b_theta_k
        b_theta_k = dgmrf(graph_b_f_k, with_bias=True) # b_theta_k = S_k @ b_f_k + b_s_k
        
        # S @ c
        S_c_k = -b_theta_k

        ## Q @ c = S^T @ S @ c

        # At time k
        graph_S_c_k = utils.new_graph(graph_y, new_x=S_c_k)
        Q_c_k = dgmrf(graph_S_c_k, transpose=True, with_bias=False) # S^T @ S @ c

        # At all times - repeat Q_c_k n_time times
        Q_c = torch.cat([Q_c_k for _ in range(n_time)], dim=0)

        ## F^T - block upper diagonal matrix with I in the diagonal anf F^T in the upper diagonal

        # Initialize F_k as an identity matrix
        F_k = torch.eye(config['n_lattice'] ** 2)

        # Multiply the F matrices of each TemporalLayer to get the final F
        for layer in temporal_model.layers:
            # You need to create the M matrix and add the identity matrix
            M = layer.create_matrix_M()
            F_layer = torch.eye(M.size(0)) + M
            F_k = torch.matmul(F_layer, F_k)  # Update F by multiplying by the new F_layer
        
        
        # F_k Transpose
        F_k_T = F_k.T

        # Fill F^T
        F_T = torch.zeros(n_time * n_nodes_spatial, n_time * n_nodes_spatial)

        # Fill F_T_big - upper bi-diagonal matrix
        for k in range(n_time):
            # Diagonal blocks with identity matrices
            idx_start = k * n_nodes_spatial
            idx_end = idx_start + n_nodes_spatial
            F_T[idx_start:idx_end, idx_start:idx_end] = torch.eye(n_nodes_spatial)

            # Superdiagonal blocks with F_T
            if k < n_time - 1:
                F_T[idx_start:idx_end, idx_end:(idx_end + n_nodes_spatial)] = F_k_T

        
        ## Omega @ mu = F^T @ Q @ c
                
        Omega_mu = torch.matmul(F_T, Q_c) 

        ## y_masked

        for k in range(n_time):
            if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                graph_y = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
            else:
                graph_y = dataset_dict["graph_y_" + str(k)]

            masked_y_k = graph_y.mask.to(torch.float32).unsqueeze(1) * graph_y.x  

            # Concatenate masked_y_k
            if k == 0:
                masked_y = masked_y_k
            else:
                masked_y = torch.cat((masked_y, masked_y_k), dim=0)

        ## RHS
                
        mean_rhs = Omega_mu + (1./utils.noise_var(config)) * masked_y

        post_mean = cg_solve(mean_rhs.unsqueeze(0), dgmrf, graph_y, config,
            config["inference_rtol"], verbose=True)[0]
    
    # graph_bias = utils.new_graph(graph_y, new_x=get_bias(dgmrf, graph_y))
    # Q_mu = -1.*dgmrf(graph_bias, transpose=True, with_bias=False) # Q@mu = -G^T@b

    # masked_y = graph_y.mask.to(torch.float32).unsqueeze(1) * graph_y.x
    # mean_rhs = Q_mu + (1./utils.noise_var(config)) * masked_y

    # if config["features"]:
    #     rhs_append = (1./utils.noise_var(config))*\
    #         graph_y.features.transpose(0,1)@masked_y

    #     mean_rhs = torch.cat((mean_rhs, rhs_append), dim=0)

    # post_mean = cg_solve(mean_rhs.unsqueeze(0), dgmrf, graph_y, config,
    #         config["inference_rtol"], verbose=True)[0]

    # if config["features"]:
    #     # CG returns posterior mean of both x and coeff., compute posterior
    #     post_mean_x = post_mean[:graph_y.num_nodes]
    #     post_mean_beta = post_mean[graph_y.num_nodes:]

    #     post_mean = post_mean_x + graph_y.features@post_mean_beta

    #     # Plot posterior mean for x alone
    #     graph_post_mean_x = utils.new_graph(graph_y, new_x=post_mean_x)
    #     vis.plot_graph(graph_post_mean_x, name="post_mean_x", title="X Posterior Mean")

    # graph_post_mean = utils.new_graph(graph_y, new_x=post_mean)

    # Posterior samples and marginal variances
    # Batch sampling
    posterior_samples_list = []
    cur_post_samples = 0
    while cur_post_samples < config["n_post_samples"]:
        posterior_samples_list.append(sample_posterior(config["n_training_samples"],
            dgmrf, graph_y, config, config["inference_rtol"], verbose=True))
        cur_post_samples += config["n_training_samples"]

    posterior_samples = torch.cat(posterior_samples_list,
           dim=0)[:config["n_post_samples"]]

    if config["features"]:
        # Include linear feature model to posterior samples
        post_samples_x = posterior_samples[:,:graph_y.num_nodes]
        post_samples_coeff = posterior_samples[:,graph_y.num_nodes:]

        posterior_samples = post_samples_x + graph_y.features@post_samples_coeff

    # MC estimate of variance using known population mean
    post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # Posterior std.-dev. for y
    post_std = torch.sqrt(post_var_x + utils.noise_var(config))

    graph_post_std = utils.new_graph(graph_y, new_x=post_std)
    graph_post_sample = utils.new_graph(graph_y)

    # Plot posterior samples
    for sample_i, post_sample in enumerate(
            posterior_samples[:config["plot_post_samples"]]):
        graph_post_sample.x = post_sample
        vis.plot_graph(graph_post_sample, name="post_sample",
                title="Posterior sample {}".format(sample_i))

    return graph_post_mean, graph_post_std

