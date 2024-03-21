import torch
import torch_geometric as ptg

from lib.cg_batch import cg_batch
import utils

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
        max_outer_iter=2, 
        max_inner_iter=3, 
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

    # # Divergence counter
    # count_divergence = 0
    
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
            
    #     # Update best solution
    #     if outer_iter == 0:
    #         best_norm_residuals = norm_residuals
    #         best_solution = x_i
    #         best_outer_iter = outer_iter
    #         best_cg_info = cg_info
    #     elif norm_residuals < best_norm_residuals and cg_info["optimal"] is True:
    #         best_norm_residuals = norm_residuals
    #         best_solution = x_i
    #         best_outer_iter = outer_iter
    #         best_cg_info = cg_info
        

    #     # Check for convergence
    #     if norm_residuals < rtol: # or cg_info["optimal"] is True:
    #         break 

    #     # Check for divergence
    #     if norm_residuals > best_norm_residuals and cg_info["optimal"] is False:
    #         count_divergence += 1
    #     else: # Reset to zero
    #         count_divergence = 0

    #     if count_divergence > 10:
    #         print("Divergence detected. Exiting before reaching the maximum number of outer iterations.")
    #         break

    # # Optionally print final CG information
    # if verbose:
    #     print("Solution with lowest residual norm was found in Outer iteration {} with residual norm {}. The inner CG had solution optimal: {}".format(
    #         best_outer_iter, best_norm_residuals, best_cg_info["optimal"]))

    return x_i


# Solve Omega_plus x = rhs using Conjugate Gradient
def cg_solve(
        rhs, 
        temporal_model, 
        dgmrf, 
        config, 
        rtol, 
        cg_start_guess,
        graph_dummy, 
        mask_stack,
        verbose=False):
    
    #### Conjugate Gradient
    #### Solve Omega_plus x = rhs, where Omega_plus = F^T @ Q @ F + (1/sigma^2) * I_masked, see equation (17)
   
    # (1) Compute F @ x 
    # (2) Compute Q @ (F @ x) = S^T @ S @ (F @ x)
    # (3) Compute F^T @ (Q @ (F @ x)) = F^T @ S^T @ S @ (F @ x)
    # (4) (1/sigma^2)masked_obs

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64) 
    cg_start_guess = cg_start_guess.to(torch.float64)

    # x = torch.randn(config["n_time"] * config["n_space"]) # random initial guess

    # Define Omega_plus as a function of x
    def Omega_plus_func(x):

        ## (1) F @ x

        # Reshape x to math the input shape of the temporal model
        x = x.reshape(config["n_time"], 1, config["n_space"]) # the second 1 is n_sample = 1. This dimension is necessary for the temporal model below

        # Run x through temporal model without bias to get F @ x 
        # f(x) = Fx + b_f = h, which is a vector with elements h_0 = x_0 + b_f_0
        # and h_k = x_{k+1} - F_k x_k + b_f_k for k = 1, ..., n-1  
        # !! WE ASSUME THAT F_k is equal for all k and that the layers in F_k are shared across time steps !!
        x_k_plus_1 = x[1:] # [1:] exclude first time step
        F_k_x_k = temporal_model(x[:-1].unsqueeze(3), with_bias=False) # [:-1] exclude last time step
        x_0 = x[0].unsqueeze(0) # [0] first time step
        F_x = torch.cat((x_0, x_k_plus_1 - F_k_x_k.squeeze(3))) 

        ## (2) Q @ (F @ x) = S^T @ S @ (F @ x)

        # Run Fx through spatial model without bias to obtain S @ (F @ x)
        # !! WE ASSUME THAT S_k is equal for all k and that the layers in S_k are shared across time steps !!
        graph_dummy.x = F_x.reshape(-1, 1)  # Reshape Fx to [n_time * n_space, 1]
        S_F_x = dgmrf(graph_dummy, with_bias=False)

        # Run S_F_x through spatial model without bias and transpose equal True to obtain Q @ F @ x = S^T (S @ F @ x)
        graph_dummy.x = S_F_x
        Q_F_x = dgmrf(graph_dummy, transpose=True, with_bias=False)


        ## (3) F^T @ (Q @ (F @ x)) = F^T @ S^T @ S @ (F @ x)

        # Reshape Q_F_x to math the input shape of the temporal model
        Q_F_x = Q_F_x.reshape(config["n_time"], 1, config["n_space"]) # the second 1 is n_sample = 1. This dimension is necessary for the temporal model below

        # Run Q_F_x through temporal model without bias and transpose equal True to obtain F^T @ (Q @ F @ x)
        x_k = Q_F_x[:-1] # [:-1] exclude last time step
        F_T_k_x_k_plus_1 = temporal_model(Q_F_x[1:].unsqueeze(3), transpose=True, with_bias=False) # [1:] exclude first time step
        x_n_minus_1 = Q_F_x[-1].unsqueeze(0) # [-1] last time step
        F_T_Q_F_x = torch.cat((x_k - F_T_k_x_k_plus_1.squeeze(3), x_n_minus_1)) 
        F_T_Q_F_x = F_T_Q_F_x.reshape(-1, 1) # shape [n_time * n_space, 1]
        
        ## (4) (1/sigma^2)masked_obs
  
        mask = mask_stack.reshape(-1,1) # shape [n_time * n_space, 1]
        noise_term = (1./utils.noise_var(config)) * mask


        ## Omega_plus x = F^T @ Q @ F @ x + (1/sigma^2) * I_masked @ x

        Omega_plus_x = F_T_Q_F_x + noise_term * x.reshape(-1, 1)

        return Omega_plus_x.to(torch.float64)

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
    dgmrf, 
    vi_dist, 
    config, 
    graph_dummy,
    obs_mask_stack,
    mask_stack,
):
    
    #### Posterior mean
    #### RHS consists of Omega @ mu + (1 / sigma ^ 2) * masked_obs, where Omega @ mu = = F^T @ Q @ c = F^T @ S^T @ S @ c, see equation (9)
   
    # (1) Compute S @ c - recall that ST-DGMRF model bias is b_theta = -(S @ c), cf. section 3.1.2
    # (2) Compute Q @ c = S^T @ S @ c
    # (3) Compute F^T - block upper diagonal matrix with I in the diagonal anf F^T in the upper diagonal
    # (4) Omega @ mu = F^T @ Q @ c
    # (5) (1/sigma^2)masked_obs
    # (6) RHS = Omega @ mu + (1/sigma^2)masked_obs

    ## (1) S @ c 

    # Feed zeros through the temporal model to obtain overall bias
    # !! WE ASSUME THAT THE BIAS IS THE SAME FOR ALL TIME POINTS !!
    zeros = torch.zeros(1, 1, config["n_space"], 1)
    b_f_k = temporal_model(zeros, with_bias=True) 

    # Feed b_f through spatial model to obtain model bias b_theta
    # !! WE ASSUME THAT THE BIAS IS THE SAME FOR ALL TIME POINTS !!
    graph_dummy.x = b_f_k
    b_theta_k = dgmrf(graph_dummy, with_bias=True) # b_theta_k = S_k @ b_f_k + b_s_k

    # S @ c = -b_theta, cf. equation (10)
    S_c_k = -b_theta_k

    ## (2) Q @ c = S^T @ S @ c

    graph_dummy.x = S_c_k
    Q_c_k = dgmrf(graph_dummy, transpose=True, with_bias=False) # S^T @ S @ c
    
    # Repeat Q_c_k for all time points
    Q_c = torch.tile(Q_c_k.squeeze(1), (config["n_time"], 1, 1)) # the second 1 is n_sample = 1. This dimension is necessary for the temporal model below


    ## (3) F^T x, where x is the vector of Q_c 

    # F^T x = y, which is a vector with elements y_k = x_k - F^T_k x_{k+1} for k = 0, ..., n-2 and y_n = x_n  
    x_k = Q_c[:-1] # [:-1] exclude last time step
    F_T_k_x_k_plus_1 = temporal_model(Q_c[1:].unsqueeze(3), transpose=True, with_bias=False) # [1:] exclude first time step
    x_n_minus_1 = Q_c[-1].unsqueeze(0) # [-1] last time step
    F_T_Q_c = torch.cat((x_k - F_T_k_x_k_plus_1.squeeze(3), x_n_minus_1)) 


    ## (4) Omega @ mu = F^T @ Q @ c

    Omega_mu = F_T_Q_c.squeeze(1).reshape(-1, 1) # shape [n_time * n_space, 1]


    ## (5) Noise obs term (1/sigma^2)masked_obs
    
    masked_obs = obs_mask_stack.reshape(-1,1) # shape [n_time * n_space, 1]
    noise_obs_term = (1./utils.noise_var(config)) * masked_obs

    ## (6) RHS = Omega_mu + noise_obs_term
            
    rhs = Omega_mu + noise_obs_term
    rhs = rhs.unsqueeze(0) # batch dimension - necessary for cg_batch function

    ## CG Solve

    # Initial guess - use mean of VI distribution
    cg_start_guess = vi_dist.mean_param.reshape(-1, 1)
    cg_start_guess = cg_start_guess.unsqueeze(0) # batch dimension - necessary for cg_batch function

    # Posterior mean
    post_mean = cg_solve(rhs=rhs,
                         temporal_model=temporal_model, 
                         dgmrf=dgmrf, 
                         config=config, 
                         rtol=config["inference_rtol"], 
                         cg_start_guess=cg_start_guess,
                         graph_dummy=graph_dummy,
                         mask_stack=mask_stack,
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

