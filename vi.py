import torch
import torch_geometric as ptg

import utils
from layers.flex import FlexLayer


class VariationalDist(torch.nn.Module):
    def __init__(self, config, graph_y):
        super().__init__()

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]
        self.n_time = config["n_time"]
        self.n_space = config["n_space"]

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(graph_y.mask*graph_y.x[:,0])
        self.diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.n_space) - 1.) # U(-1,1)

        self.layers = torch.nn.ModuleList([FlexLayer(graph_y, config, vi_layer=True)
                                           for _ in range(config["vi_layers"])])
        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.n_space) - 1.)

        # Reuse same batch with different x-values
        self.sample_batch = ptg.data.Batch.from_data_list([utils.new_graph(graph_y)
                                    for _ in range(self.n_samples)])

        if config["features"]:
            # Additional variational distribution for linear coefficients
            n_features = graph_y.features.shape[1]
            self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(n_features))
            self.coeff_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(n_features) - 1.) # U(-1,1)

            self.coeff_inv_std = config["coeff_inv_std"]

    @property
    def std(self):
        # Note: Only std before layers
        return torch.nn.functional.softplus(self.diag_param)

    @property
    def post_diag(self):
        # Diagonal of diagonal matrix applied after layers
        return torch.nn.functional.softplus(self.post_diag_param)

    @property
    def coeff_std(self):
        return torch.nn.functional.softplus(self.coeff_diag_param)

    def sample(self):
        standard_sample = torch.randn(self.n_samples, self.n_space)
        ind_samples = self.std * standard_sample

        self.sample_batch.x = ind_samples.reshape(-1,1)  # Stack all
        print("x: " + str(self.sample_batch.x.shape))
        print("edge_index: " + str(self.sample_batch.edge_index.shape))

        for layer in self.layers:
            propagated = layer(self.sample_batch.x, self.sample_batch.edge_index,
                            transpose=False, with_bias=False)
            self.sample_batch.x = propagated

        samples = self.sample_batch.x.reshape(self.n_samples, -1)
        
        if self.layers:
            # Apply post diagonal matrix
            samples = self.post_diag * samples

        samples = samples + self.mean_param  # Add mean last (not changed by layers)

        return samples  # shape (n_samples, n_nodes)

    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        standard_sample = torch.randn(n_samples, self.coeff_mean_param.shape[0])
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # shape (n_samples, n_features)

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, graph_y, config):
        # Compute mean and marginal std of distribution (posterior estimate)
        # Mean
        graph_post_mean = utils.new_graph(graph_y,
                new_x=self.mean_param.detach().unsqueeze(1))

        # Marginal std. (MC estimate)
        mc_sample_list = []
        cur_mc_samples = 0
        while cur_mc_samples < config["n_post_samples"]:
            mc_sample_list.append(self.sample())
            cur_mc_samples += self.n_samples
        mc_samples = torch.cat(mc_sample_list, dim=0)[:config["n_post_samples"]]

        # MC estimate of variance using known population mean
        post_var_x = torch.mean(torch.pow(mc_samples - self.mean_param, 2), dim=0)
        # Posterior std.-dev. for y
        post_std = torch.sqrt(post_var_x + utils.noise_var(config)).unsqueeze(1)

        graph_post_std = utils.new_graph(graph_y, new_x=post_std)

        return graph_post_mean, graph_post_std


class VariationalDistBatch(torch.nn.Module):
    def __init__(self, config, graph_y, mean_param):
        super().__init__()

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]
        self.n_time = config["n_time"]
        self.n_space = config["n_space"]
        self.graph_y = graph_y

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(mean_param)
        self.diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.n_space * self.n_time) - 1.) # U(-1,1)

        # Share layers across time. Hence, only create config["vi_layers"] layers
        self.layers = torch.nn.ModuleList([FlexLayer(graph_y, config, vi_layer=True)
                                           for _ in range(config["vi_layers"])])

        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.n_space * self.n_time) - 1.)

        # Reuse same batch with different x-values
        self.sample_batch = ptg.data.Batch.from_data_list([utils.new_graph(graph_y)
                                    for _ in range(self.n_samples * self.n_time)])

    @property
    def std(self):
        # Note: Only std before layers
        return torch.nn.functional.softplus(self.diag_param)

    @property
    def post_diag(self):
        # Diagonal of diagonal matrix applied after layers
        return torch.nn.functional.softplus(self.post_diag_param)

    def sample(self):
        # Generate a batch of standard normal samples with an additional time dimension
        standard_sample = torch.randn(self.n_time, self.n_samples, self.n_space)

        # Apply the variational distribution's standard deviation, reshaping as needed
        ind_samples = self.std * standard_sample.reshape(self.n_samples, self.n_space * self.n_time)

        # Flatten ind_samples for batch processing across time steps
        ind_samples_batch_format = ind_samples.reshape(-1, 1)

        # Set the x-values of the batch to the individual samples
        self.sample_batch.x = ind_samples_batch_format

        for layer in self.layers:
            propagated = layer(self.sample_batch.x, self.sample_batch.edge_index,
                            transpose=False, with_bias=False)
            self.sample_batch.x = propagated # Shape (n_space*n_graphs,1) where n_graphs = n_samples*n_time
        
        if self.layers:
            # Apply post diagonal matrix, reshaping as needed
            samples = self.post_diag * self.sample_batch.x.reshape(self.n_samples, -1)

        # Reshape back to start shape (n_time, n_samples, n_space)
        samples = self.sample_batch.x.reshape(self.n_time, self.n_samples, self.n_space)

        # Reshape mean_param to match samples for correct broadcasting
        mean_param_reshaped = self.mean_param.view(self.n_time, 1, self.n_space)
        samples += mean_param_reshaped # Add mean last (not changed by layers)

        return samples  # shape (n_time, n_samples, n_space)
    
    def log_det(self):
        # Assuming the log determinants from layers are the same for all time steps
        layers_log_det = sum([layer.log_det() for layer in self.layers])

        # Reshape std and post_diag to separate time and space dimensions
        # Then compute the log det for each time step and sum across all space dimensions
        # Finally, sum across all time steps
        std_log_det = torch.sum(torch.log(self.std.view(self.n_time, self.n_space)), dim=1)
        total_std_log_det = torch.sum(std_log_det)

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag.view(self.n_time, self.n_space)), dim=1)
            total_post_diag_log_det = torch.sum(post_diag_log_det)

        # Combine them with the layers' log det, considering the factor of 2
        total_log_det = 2.0 * (total_std_log_det + layers_log_det + total_post_diag_log_det)

        return total_log_det

    
