##### TEST #####

vi_dist_list = [0]
vi_dist_list[0] = vi.VariationalDist(config, graph_k)
dgmrf_list = [0]
dgmrf_list[0] = dgmrf

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
            vi_samples_k = vi_samples[k] #vi_dist_k.sample()

            # Prepare vi samples for batched temporal model
            # Hence, reshape to (n_samples, n_nodes, 1) for torch.bmm in temporal.py to work correctly
            vi_samples_temporal_batch_format = vi_samples_k.unsqueeze(2)

            # Feed samples through temporal model
            h_k = temporal_model(vi_samples_temporal_batch_format)

            # Prepare vi samples after temporal transform for batched spatial model
            vi_dist_k.sample_batch.x = h_k.reshape(-1,1)

            # Feed samples through spatial model - SHOULD WE HAVE A DGMRF FOR EACH TIME STEP - dgmrf_k, see section 3.2.1??
            dgmrf_k = dgmrf_list[k]
            g_k = dgmrf_k(vi_dist_k.sample_batch)

            torch.allclose(g_k, g)

            # Compute log determinant of variational distribution
            vi_log_det1 = vi_dist_k.log_det()

            vi_dist_k.layers[0].log_det()
            vi_dist.layers[0].log_det()

            for name, param in vi_dist_k.named_parameters():
                print(name, param.data)

            for name, param in vi_dist.named_parameters():
                print(name, param.data)

            param_name = 'diag_param'  # Example parameter name
            param_value1 = vi_dist_k.state_dict()[param_name]
            param_value = vi_dist.state_dict()[param_name]
            torch.allclose(param_value1, param_value)

            # Compute ELBO components for time step k
            l1 = 0.5 * vi_log_det
            l2_test = -graph_k.n_observed * config["log_noise_std"]
            l3_test = dgmrf_k.log_det()
            l4_test = -(1./(2. * config["n_training_samples"])) * torch.sum(torch.pow(g_k,2))
            l5_test = -(1./(2. * utils.noise_var(config)*\
                config["n_training_samples"])) * torch.sum(torch.pow(
                    (vi_samples_k - graph_k.x.flatten()), 2)[:, graph_k.mask])
            
            z1 = torch.pow((vi_samples_k - graph_k.x.flatten()), 2)[:, graph_k.mask]
            z1.shape
            z = torch.pow((vi_samples - obs_stack.unsqueeze(1)), 2)[:,:, graph_k.mask]
            z.shape

            torch.allclose(z1, z)
            
            # Update ELBO
            elbo += l1 + l2 + l3 + l4 + l5