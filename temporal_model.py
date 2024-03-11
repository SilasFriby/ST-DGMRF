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

    def forward(self, x, batch_run=True, n_samples=None):
        # Sequentially apply each layer to the data
        for layer in self.layers:
                x = layer(x, batch_run, n_samples) 
        return x


