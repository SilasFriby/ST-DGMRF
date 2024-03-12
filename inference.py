import torch
import torch_geometric as ptg

from lib.cg_batch import cg_batch
import utils
import visualization as vis

def get_bias(dgmrf, graph_y):
    zero_graph = utils.new_graph(graph_y, new_x=torch.zeros(graph_y.num_nodes, 1))
    bias = dgmrf(zero_graph)
    return bias

# Regularized CG from section B.4
def regularized_cg(A_func, B, x0, nu0=10.0, rtol=1e-7, max_outer_iter=100, max_inner_iter=200, nu_decay_factor=10.0, verbose=False):
    
    # Initial guess
    x = x0 
    nu = nu0

    ## Solve the regularized linear system (νI + A)x = νx^(i) + b
    
    # Left-hand side
    def lhs(x, nu):
        return nu * x + A_func(x)

    # Right-hand side        
    def rhs(x, nu): 
        return nu * x + B  
    
    # Run the regularized CG iterations
    for outer_iter in range(max_outer_iter):

        # Use the standard CG method to solve the regularized system
        solution, cg_info = cg_batch(A_bmm=lambda x: lhs(x, nu = nu), B=rhs(x, nu), rtol=rtol, maxiter=max_inner_iter, verbose=verbose)
        
        # Optionally print CG information
        if verbose:
            print("Outer iteration {}: CG finished in {} iterations, solution optimal: {}".format(
                outer_iter, cg_info["niter"], cg_info["optimal"]))
        
        # Update x
        x = solution

        # Decrease nu every 10 iterations by the decay factor
        if outer_iter % 10 == 0:
            nu /= nu_decay_factor
            
        # Check for convergence or max iterations reached
        residuals = lhs(x=solution, nu=nu) - rhs(x=solution, nu=nu)  
        if torch.norm(residuals) < rtol:
            break   

    return x


# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, temporal_model, dgmrf, graph_y, config, n_time, dataset_dict, rtol, cg_start_guess, verbose=False):
    
    # Number of nodes
    n_nodes_spatial = graph_y.num_nodes

    # Initialize graph
    graph_k = utils.new_graph(graph_y, new_x=torch.zeros(n_nodes_spatial, 1))

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)    

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, n_nodes, 1)
        # Implicitly applies posterior precision matrix Q_tilde to a vector x
        # y = (F^T @ S^T @ S @ F + (1/sigma^2)*I_masked)x
        # (1) Create Q_F_x = (S^T @ S @ F)x
        # (2) Create F^T as sparse matrix
        # (3) y first term: (F^T @ S^T @ S @ F)x - using spare matrix multiplication
        # (4) y second term: (1/sigma^2)I_masked
        # (5) Result


        ## (1) Create Q_F_x = (S^T @ S @ F)x

        # Reshape x to the desired shape from [1, n_time * n_nodes_spatial, 1] to [n_time, n_nodes_spatial, 1]
        x_matrix = x.view(n_time, n_nodes_spatial, 1)

        # Loop over time
        for k in range(n_time):

            x_k = x_matrix[k,:,:].unsqueeze(0)

            # Feed x_matrix through temporal model to get F @ x_k. 
            # Note that we set n_samples equal to columns in x_matrix. Hence, F_x_k is computed in a batch manner instead of looping over time 
            F_x_k = temporal_model(x_k, with_bias=False, overwrite_n_samples=x_k.shape[0])

            # Feed F_k_x through spatial model to obtain S_k @ F_k @ x_k
            graph_k.x = F_x_k.squeeze(0)
            S_F_x_k = dgmrf(graph_k, with_bias=False)

            # Feed S_F_x_k through spatial model with transpose=True to obtain S^T_k @ F_k @ S_k @ x_k = Q_k @ F_k @ x_k
            graph_k.x = S_F_x_k
            Q_F_x_k = dgmrf(graph_k, transpose=True, with_bias=False)

            # Concatenate Q_F_x_k
            if k == 0:
                Q_F_x = Q_F_x_k
            else:
                Q_F_x = torch.cat((Q_F_x, Q_F_x_k), dim=0)

        ## (2) Create F^T as sparse matrix - block upper diagonal matrix with I in the diagonal anf F_k^T in the upper diagonal

        # Initialize F_k as an identity matrix
        F_k = torch.eye(n_nodes_spatial)

        # Multiply the F matrices of each TemporalLayer to get the final F - !!!!WORKS ONLY FOR LAYER I - M!!!!
        for layer in temporal_model.layers:
            # You need to create the M matrix and add the identity matrix
            M = layer.create_matrix_M()
            F_layer = torch.eye(M.size(0)) + M
            F_k = torch.matmul(F_layer, F_k)  # Update F by multiplying by the new F_layer
        
        # F_k Transpose
        F_k_T = torch.transpose(F_k, 0, 1)

        # Initialize lists for indices and values for the sparse tensor
        indices_list = []
        values_list = []

        for i in range(n_time):
            # Identity block indices and values
            row_indices = torch.arange(i * n_nodes_spatial, (i + 1) * n_nodes_spatial)
            col_indices = row_indices  # Same for identity matrix
            id_indices = torch.stack([row_indices, col_indices], dim=0)
            
            # Values for the identity matrix are all ones
            id_values = torch.ones(n_nodes_spatial)
            
            indices_list.append(id_indices)
            values_list.append(id_values)
            
            # Add F_k_T on the superdiagonal
            if i < n_time - 1:
                # Superdiagonal block indices
                super_row_indices = torch.arange(i * n_nodes_spatial, (i + 1) * n_nodes_spatial).unsqueeze(1).repeat(1, n_nodes_spatial).flatten()
                super_col_indices = torch.arange((i + 1) * n_nodes_spatial, (i + 2) * n_nodes_spatial).repeat(n_nodes_spatial)
                super_indices = torch.stack([super_row_indices, super_col_indices], dim=0)
                
                # Flatten F_k_T_example to get the values for the superdiagonal block
                super_values = F_k_T.flatten()
                
                indices_list.append(super_indices)
                values_list.append(super_values)

        # Concatenate indices and values from all blocks
        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)

        # Create the sparse tensor representing F_T
        size = (n_time * n_nodes_spatial, n_time * n_nodes_spatial)
        F_T = torch.sparse_coo_tensor(all_indices, all_values, size)

        # print(F_T.to_dense())

        ## (3) y first term: (F^T @ S^T @ S @ F)x - using spare matrix multiplication

        F_T = F_T.to(torch.float64) # Convert to float64 to match Q_F_x dtype
        y_first_term = torch.sparse.mm(F_T, Q_F_x)
        y_first_term = y_first_term.unsqueeze(0)


        ## (4) y second term: (1/sigma^2)y_masked

        for k in range(n_time):
            if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
                graph_y = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
            else:
                graph_y = dataset_dict["graph_y_" + str(k)]

            mask_k = graph_y.mask.to(torch.float64).unsqueeze(1)  

            # Concatenate masked_y_k
            if k == 0:
                mask = mask_k
            else:
                mask = torch.cat((mask, mask_k), dim=0)

        y_second_term = (1./utils.noise_var(config)) * mask
        y_second_term = y_second_term.unsqueeze(0)


        ## (5) Result

        y = y_first_term + y_second_term * x
        
        # Return result
        return y
    
    ## Regularized CG

    # solution, cg_info = cg_batch(A_bmm=Q_tilde_batched, B=rhs, rtol=rtol, verbose=verbose)
    solution = regularized_cg(A_func=Q_tilde_batched, B=rhs, x0=cg_start_guess, rtol = rtol, verbose=verbose)

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
def posterior_inference(temporal_model, dgmrf, vi_dist, config, graph_y, n_time, dataset_dict):
    
    #### Posterior mean
    #### RHS consists of Omega @ mu + (1 / sigma ^ 2) * y_masked, where Omega @ mu = = F^T @ Q @ c = F^T @ S^T @ S @ c
   
    # (1) Compute S @ c - recall that ST-DGMRF model bias is b_theta = -(S @ c), cf. section 3.1.2
    # (2) Compute Q @ c = S^T @ S @ c
    # (3) Compute F^T - block upper diagonal matrix with I in the diagonal anf F^T in the upper diagonal
    # (4) (1/sigma^2)y_masked

    ## (1) S @ c 
    
    # Feed zeros through temporal model to get temporal bias b_f at time k
    n_nodes_spatial = graph_y.num_nodes
    zeros_temporal_format = torch.zeros(1, n_nodes_spatial, 1) 
    b_f_k = temporal_model(zeros_temporal_format, overwrite_n_samples=1)

    # Prepare b_f_k for spatial model
    graph_k = utils.new_graph(graph_y, new_x=b_f_k.squeeze(0))

    # Feed graph_b_f_k through spatial model to obtain model bias at time k b_theta_k
    b_theta_k = dgmrf(graph_k, with_bias=True) # b_theta_k = S_k @ b_f_k + b_s_k
    S_c_k = -b_theta_k

    ## (2) Q @ c = S^T @ S @ c

    # At time k
    graph_k.x = S_c_k
    Q_c_k = dgmrf(graph_k, transpose=True, with_bias=False) # S^T @ S @ c

    # At all times - repeat Q_c_k n_time times. !!!!ONLY CORRECT IF c = (mu_0, c_1, ..., c_K) IS CONSTANT OVER TIME!!!!
    Q_c = torch.cat([Q_c_k for _ in range(n_time)], dim=0)

    ## (3) F^T - block upper diagonal matrix with I in the diagonal anf F^T in the upper diagonal

    # Initialize F_k as an identity matrix
    F_k = torch.eye(n_nodes_spatial)

    # Multiply the F matrices of each TemporalLayer to get the final F - !!!! WORKS ONLY FOR LAYER I - M!!!!
    for layer in temporal_model.layers:
        # You need to create the M matrix and add the identity matrix
        M = layer.create_matrix_M()
        F_layer = torch.eye(M.size(0)) + M
        F_k = torch.matmul(F_layer, F_k)  # Update F by multiplying by the new F_layer
    
    
    # F_k Transpose
    F_k_T = torch.transpose(F_k, 0, 1)

    # Initialize lists for indices and values for the sparse tensor
    indices_list = []
    values_list = []

    for i in range(n_time):
        # Identity block indices and values
        row_indices = torch.arange(i * n_nodes_spatial, (i + 1) * n_nodes_spatial)
        col_indices = row_indices  # Same for identity matrix
        id_indices = torch.stack([row_indices, col_indices], dim=0)
        
        # Values for the identity matrix are all ones
        id_values = torch.ones(n_nodes_spatial)
        
        indices_list.append(id_indices)
        values_list.append(id_values)
        
        # Add F_k_T on the superdiagonal
        if i < n_time - 1:
            # Superdiagonal block indices
            super_row_indices = torch.arange(i * n_nodes_spatial, (i + 1) * n_nodes_spatial).unsqueeze(1).repeat(1, n_nodes_spatial).flatten()
            super_col_indices = torch.arange((i + 1) * n_nodes_spatial, (i + 2) * n_nodes_spatial).repeat(n_nodes_spatial)
            super_indices = torch.stack([super_row_indices, super_col_indices], dim=0)
            
            # Flatten F_k_T_example to get the values for the superdiagonal block
            super_values = F_k_T.flatten()
            
            indices_list.append(super_indices)
            values_list.append(super_values)

    # Concatenate indices and values from all blocks
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)

    # Create the sparse tensor representing F_T
    size = (n_time * n_nodes_spatial, n_time * n_nodes_spatial)
    F_T = torch.sparse_coo_tensor(all_indices, all_values, size)

    
    ## Omega @ mu = F^T @ Q @ c
            
    Omega_mu = torch.sparse.mm(F_T, Q_c)

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

    ## CG Solve

     # Initial guess - use mean of VI distribution. Note the r=use of 'repeat' is ONLY CORRECT WHEN VI IS TIME-INVARIANT !!!!
    cg_start_guess_batch = vi_dist.mean_param.repeat(n_time).unsqueeze(0).unsqueeze(2) # unsqueeze dimensions necessary for cg_batch function
    
    # Posterior mean
    post_mean = cg_solve(rhs = mean_rhs.unsqueeze(0), 
                         temporal_model=temporal_model, 
                         dgmrf=dgmrf, 
                         graph_y=graph_y, 
                         config=config, 
                         n_time=n_time, 
                         dataset_dict=dataset_dict, 
                         rtol=config["inference_rtol"], 
                         cg_start_guess=cg_start_guess_batch,
                         verbose=True)[0]
    
    # graph_post_mean = utils.new_graph(graph_y, new_x=post_mean)

    # # Posterior samples and marginal variances
    # # Batch sampling
    # posterior_samples_list = []
    # cur_post_samples = 0
    # while cur_post_samples < config["n_post_samples"]:
    #     posterior_samples_list.append(sample_posterior(config["n_training_samples"],
    #         dgmrf, graph_y, config, config["inference_rtol"], verbose=True))
    #     cur_post_samples += config["n_training_samples"]

    # posterior_samples = torch.cat(posterior_samples_list,
    #        dim=0)[:config["n_post_samples"]]

    # if config["features"]:
    #     # Include linear feature model to posterior samples
    #     post_samples_x = posterior_samples[:,:graph_y.num_nodes]
    #     post_samples_coeff = posterior_samples[:,graph_y.num_nodes:]

    #     posterior_samples = post_samples_x + graph_y.features@post_samples_coeff

    # # MC estimate of variance using known population mean
    # post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # # Posterior std.-dev. for y
    # post_std = torch.sqrt(post_var_x + utils.noise_var(config))

    # graph_post_std = utils.new_graph(graph_y, new_x=post_std)
    # graph_post_sample = utils.new_graph(graph_y)

    # # Plot posterior samples
    # for sample_i, post_sample in enumerate(
    #         posterior_samples[:config["plot_post_samples"]]):
    #     graph_post_sample.x = post_sample
    #     vis.plot_graph(graph_post_sample, name="post_sample",
    #             title="Posterior sample {}".format(sample_i))

    return post_mean #graph_post_mean, graph_post_std

