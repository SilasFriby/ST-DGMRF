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
def regularized_cg(
        A_func, 
        B, 
        x0, 
        rtol,
        nu0=10.0,  
        max_outer_iter=100, 
        max_inner_iter=200, 
        nu_decay_factor=10.0, 
        verbose=False):
    
    # Initial values
    x_i = x0 
    nu = nu0

    ## Solve the regularized linear system (νI + A)x = νx^(i) + b
    
    # Left-hand side
    def lhs(x, nu):
        # Function cg_batch requires a batch dimension, but A_func does not
        # so here we remove the batch dimension and add it again
        x_unbatched = x.squeeze(0)
        A = A_func(x_unbatched)
        A_batched = A.unsqueeze(0)
        return nu * x + A_batched

    # Right-hand side        
    def rhs(x, nu): 
        return nu * x + B    
    
    # Run the regularized CG iterations
    for outer_iter in range(max_outer_iter):

        # Use the standard CG method to solve the regularized system
        solution, cg_info = cg_batch(A_bmm=lambda x: lhs(x, nu = nu), 
                                     B=rhs(x=x_i, nu=nu), 
                                     rtol=rtol, 
                                     maxiter=max_inner_iter)
        
       
        # Update x
        x_i = solution

        # Decrease nu every 10 iterations by the decay factor
        if outer_iter > 0 and outer_iter % 10 == 0:
            nu /= nu_decay_factor
            
        # Norm of residuals
        residuals = lhs(x=x_i, nu=nu) - rhs(x=x_i, nu=nu)  
        norm_residuals = torch.norm(residuals)

         # Optionally print CG information
        if verbose:
            print("Outer iteration {}: Outer residual norm {}: Inner CG finished in {} iterations, Inner CG solution optimal: {}".format(
                outer_iter, norm_residuals, cg_info["niter"], cg_info["optimal"]))
        

        # Check for convergence
        if norm_residuals < rtol: # or cg_info["optimal"] is True:
            break   

    return x_i


# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(
        rhs, 
        temporal_model, 
        dgmrf_list, 
        config, 
        dataset_dict, 
        rtol, 
        cg_start_guess, 
        verbose=False):
    
    n_time = config["n_time"]
    n_space = config["n_space"]

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64) 
    cg_start_guess = cg_start_guess.to(torch.float64)   

    ## Create Q as sparse matrix - diagonal block matrix with Q_k in every block

    # Initialize lists for indices and values for the sparse tensor
    indices_list = []
    values_list = []

    for k in range(n_time):
        # Q_k block indices and values
        row_indices = torch.arange(k * n_space, (k + 1) * n_space).unsqueeze(1).repeat(1, n_space).flatten()
        col_indices = torch.arange(k * n_space, (k + 1) * n_space).repeat(n_space)
        id_indices = torch.stack([row_indices, col_indices], dim=0)

        ## Q_k = S_k^T @ S_k

        # Spatial model at time k
        dgmrf_k = dgmrf_list[k]

        # Initialize S_k as an identity matrix
        S_k = torch.eye(n_space)

        # Multiply the S matrices of each spatial layer to get the final S_k - !!!!ONLY CORRECT IF THERE ARE NO NON-LINEAR LAYERS!!!!
        for layer in dgmrf_k.layers:
            S_layer = layer.create_S_matrix(degree_matrix=dataset_dict['degree_matrix'], 
                                            adjacency_matrix=dataset_dict['adjacency_matrix'])
            S_k = torch.matmul(S_layer, S_k)  # Update F by multiplying by the new F_layer

        # S_k Transpose
        S_k_T = torch.transpose(S_k, 0, 1)

        # Q_k
        Q_k = torch.matmul(S_k_T, S_k)

        # Values for the Q_k block
        id_values = Q_k.flatten()

        indices_list.append(id_indices)
        values_list.append(id_values)

    # Concatenate indices and values from all blocks
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)

    # Create the sparse tensor representing Q
    size = (n_time * n_space, n_time * n_space)
    Q = torch.sparse_coo_tensor(all_indices, all_values, size)
    # print(Q.to_dense())

    ## Create F and F^T as sparse matrices
    ## F is a block lower bi-diagonal matrix with I in the diagonal and -F_k in the subdiagonal
    ## F^T is a block upper bi-diagonal matrix with I in the diagonal and -F_k^T in the superdiagonal

    # Initialize F_k as an identity matrix
    F_k = torch.eye(n_space)

    # Multiply the F matrices of each TemporalLayer to get the final F_k - !!!!WORKS ONLY FOR LAYER I - M!!!!
    for layer in temporal_model.layers:
        # You need to create the M matrix and add the identity matrix
        M_layer = layer.create_matrix_M()
        F_layer = torch.eye(M_layer.size(0)) + M_layer
        F_k = torch.matmul(F_layer, F_k)  # Update F by multiplying by the new F_layer

    # F_k Transpose
    F_k_T = torch.transpose(F_k, 0, 1)

    # Initialize lists for indices and values for the sparse tensor
    indices_list = []
    values_list = []
    indices_list_T = []
    values_list_T = []

    for i in range(n_time):
        # Identity block indices and values
        row_indices = torch.arange(i * n_space, (i + 1) * n_space)
        col_indices = row_indices  # Same for identity matrix
        id_indices = torch.stack([row_indices, col_indices], dim=0)
        
        # Values for the identity matrix are all ones
        id_values = torch.ones(n_space)
        
        indices_list.append(id_indices)
        values_list.append(id_values)
        indices_list_T.append(id_indices)
        values_list_T.append(id_values)
        
        # Add F_k on the subdiagonal and F_k_T on the superdiagonal
        if i < n_time - 1:
            # Sub- and superdiagonal block indices
            sub_row_indices = torch.arange((i + 1) * n_space, (i + 2) * n_space).unsqueeze(1).repeat(1, n_space).flatten()
            sub_col_indices = torch.arange(i * n_space, (i + 1) * n_space).repeat(n_space)
            sub_indices = torch.stack([sub_row_indices, sub_col_indices], dim=0)

            super_row_indices = torch.arange(i * n_space, (i + 1) * n_space).unsqueeze(1).repeat(1, n_space).flatten()
            super_col_indices = torch.arange((i + 1) * n_space, (i + 2) * n_space).repeat(n_space)
            super_indices = torch.stack([super_row_indices, super_col_indices], dim=0)
            
            # Flatten F_k and F_k_T to get the values for the sub- and superdiagonal block
            super_values = -F_k_T.flatten()
            sub_values = -F_k.flatten()
            
            indices_list.append(sub_indices)
            values_list.append(sub_values)
            indices_list_T.append(super_indices)
            values_list_T.append(super_values)

    # Concatenate indices and values from all blocks
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)
    all_indices_T = torch.cat(indices_list_T, dim=1)
    all_values_T = torch.cat(values_list_T)

    # Create the sparse tensor representing F and F_T
    size = (n_time * n_space, n_time * n_space)
    F = torch.sparse_coo_tensor(all_indices, all_values, size)
    F_T = torch.sparse_coo_tensor(all_indices_T, all_values_T, size)

    ## Create noise term = (1/sigma^2) * I_masked

    # Loop over time to find unobserved nodes at each time - i.e the mask
    for k in range(n_time):
        if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
            graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
        else:
            graph_k = dataset_dict["graph_y_" + str(k)]

        mask_k = graph_k.mask.to(torch.float64).unsqueeze(1)  

        # Concatenate masked_y_k
        if k == 0:
            mask = mask_k
        else:
            mask = torch.cat((mask, mask_k), dim=0)

    # Create noise term
    noise_term = (1./utils.noise_var(config)) * mask

    # Create noise_term_matrix as a sparse diag matrix with noise_term as the diagonal
    noise_term_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([torch.arange(n_time * n_space), torch.arange(n_time * n_space)]), 
        values=noise_term.flatten(), 
        size=(n_time * n_space, n_time * n_space)
    )

    ##  Create linear function for Omega plus, see equation (17)
    
    # Convert all matrices to double precision
    F_T = F_T.to(torch.float64)
    F = F.to(torch.float64)
    Q = Q.to(torch.float64)
    noise_term_matrix = noise_term_matrix.to(torch.float64)

    # Define Omega_plus as a function
    # It is more efficient to perform multiple sparse matrix multiplications with vector x, than to form the full matrix Omega_plus
    def Omega_plus_func(x):
        Omega_plus_x = torch.sparse.mm(F_T, torch.sparse.mm(Q, torch.sparse.mm(F, x))) + torch.sparse.mm(noise_term_matrix, x)
        return Omega_plus_x


    ## Regularized CG

    solution = regularized_cg(A_func=Omega_plus_func, 
                              B=rhs, 
                              x0=cg_start_guess, 
                              rtol=rtol, 
                              verbose=verbose)

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
def posterior_inference(
    temporal_model, 
    dgmrf_list, 
    vi_dist_list, 
    config, 
    dataset_dict
):
    
    #### Posterior mean
    #### RHS consists of Omega @ mu + (1 / sigma ^ 2) * y_masked, where Omega @ mu = = F^T @ Q @ c = F^T @ S^T @ S @ c, see equation (9)
   
    # (1) Compute S @ c - recall that ST-DGMRF model bias is b_theta = -(S @ c), cf. section 3.1.2
    # (2) Compute Q @ c = S^T @ S @ c
    # (3) Compute F^T - block upper diagonal matrix with I in the diagonal anf F^T in the upper diagonal
    # (4) (1/sigma^2)y_masked

    n_time = config["n_time"]
    n_space = config["n_space"]

    ## (1) S @ c 

    for k in range(n_time):
        # Load graph at time k
        if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
            graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
        else:
            graph_k = dataset_dict["graph_y_" + str(k)]

        # Feed zeros through temporal model to get temporal bias b_f at time k
        zeros_temporal_format = torch.zeros(1, n_space, 1) 
        b_f_k = temporal_model(zeros_temporal_format, with_bias=True, overwrite_n_samples=1) #  SHOULD b_f_k BE DIFFERENT FOR EACH TIME POINT?? 

        # Prepare b_f_k for spatial model
        graph_k.x = b_f_k.squeeze(0)

        # Feed b_f_k through spatial model to obtain model bias at time k b_theta_k
        dgmrf_k = dgmrf_list[k]
        b_theta_k = dgmrf_k(graph_k, with_bias=True) # b_theta_k = S_k @ b_f_k + b_s_k
        
        # S @ c = -b_theta, cf. equation (10)
        S_c_k = -b_theta_k

        ## (2) Q @ c = S^T @ S @ c

        # At time k
        graph_k.x = S_c_k
        Q_c_k = dgmrf_k(graph_k, transpose=True, with_bias=False) # S^T @ S @ c

        # Append Q_c_k
        if k == 0:
            Q_c = Q_c_k
        else: # append
            Q_c = torch.cat([Q_c, Q_c_k], dim=0)

    ## (3) F^T - block upper diagonal matrix with I in the diagonal anf -F_k^T in the super diagonal

    # Initialize F_k as an identity matrix
    F_k = torch.eye(n_space)

    # Multiply the F_layer matrices of each TemporalLayer to get the final F_k - !!!! CURRENT CODE WORKS ONLY FOR LAYER I + M!!!!
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
        row_indices = torch.arange(i * n_space, (i + 1) * n_space)
        col_indices = row_indices  # Same for identity matrix
        id_indices = torch.stack([row_indices, col_indices], dim=0)
        
        # Values for the identity matrix are all ones
        id_values = torch.ones(n_space)
        
        indices_list.append(id_indices)
        values_list.append(id_values)
        
        # Add F_k_T on the superdiagonal
        if i < n_time - 1:
            # Superdiagonal block indices
            super_row_indices = torch.arange(i * n_space, (i + 1) * n_space).unsqueeze(1).repeat(1, n_space).flatten()
            super_col_indices = torch.arange((i + 1) * n_space, (i + 2) * n_space).repeat(n_space)
            super_indices = torch.stack([super_row_indices, super_col_indices], dim=0)
            
            # Flatten F_k_T_example to get the values for the superdiagonal block
            super_values = -F_k_T.flatten()
            
            indices_list.append(super_indices)
            values_list.append(super_values)

    # Concatenate indices and values from all blocks
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)

    # Create the sparse tensor representing F_T
    size = (n_time * n_space, n_time * n_space)
    F_T = torch.sparse_coo_tensor(all_indices, all_values, size)
    # print(F_T.to_dense())

    ## Omega @ mu = F^T @ Q @ c

    F_T = F_T.to(torch.float64) 
    Q_c = Q_c.to(torch.float64)       
    Omega_mu = torch.sparse.mm(F_T, Q_c)

    ## y_masked

    for k in range(n_time):
        if config["sample_times_start"] is not None and config["sample_times_end"] is not None:
            graph_k = dataset_dict["graph_y_" + str(k + config["sample_times_start"])]
        else:
            graph_k = dataset_dict["graph_y_" + str(k)]

        masked_y_k = graph_k.mask.to(torch.float64).unsqueeze(1) * graph_k.x  

        # Concatenate masked_y_k
        if k == 0:
            masked_y = masked_y_k
        else:
            masked_y = torch.cat((masked_y, masked_y_k), dim=0)

    ## RHS
            
    rhs = Omega_mu + (1./utils.noise_var(config)) * masked_y

    # Add batch dimension - necessary for cg_batch function
    rhs = rhs.unsqueeze(0)

    ## CG Solve

    # Initial guess - use mean of VI distribution
    cg_start_guess = []

    for vi_dist in vi_dist_list:
        # Assume vi_dist.mean_param is a tensor; append it to the list
        cg_start_guess.append(vi_dist.mean_param)
    
    # Concatenate the list of tensors to a single tensor
    cg_start_guess = torch.cat(cg_start_guess, dim=0)

    # Add batch dimension - necessary for cg_batch function
    cg_start_guess = cg_start_guess.unsqueeze(0).unsqueeze(2)

    # Posterior mean
    post_mean = cg_solve(rhs = rhs,
                         temporal_model=temporal_model, 
                         dgmrf_list=dgmrf_list, 
                         config=config, 
                         dataset_dict=dataset_dict, 
                         rtol=config["inference_rtol"], 
                         cg_start_guess=cg_start_guess,
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

