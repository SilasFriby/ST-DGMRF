import torch
from layers.temporal import TemporalLayer

class TemporalModel(torch.nn.Module):
    def __init__(self, config):
        super(TemporalModel, self).__init__()
        # Construct temporal layers
        layer_list = []
        for _ in range(config['n_layers_temporal']):
            layer_list.append(TemporalLayer(config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x, transpose=False, with_bias=True):
        # x is now expected to have shape [n_time, n_samples, n_space, other_dim]
        
        # Process all time steps together through each layer
        for layer in self.layers:
            # You'll need to collapse the time and sample dimensions to treat each time step as a separate sample
            n_time, n_samples, n_space, other_dim = x.shape
            x = x.view(n_time * n_samples, n_space, other_dim)  # Collapse time and samples together

            x = layer(x, with_bias, transpose)  # Apply the layer

            # After processing, reshape x to split time and samples again
            x = x.view(n_time, n_samples, n_space, other_dim)

        return x
   


