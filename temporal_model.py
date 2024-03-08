import torch.nn as nn
from layers.temporal import TemporalLayer

class TemporalModel(nn.Module):
    def __init__(self, config):
        super(TemporalModel, self).__init__()

        # Construct temporal layers
        layer_list = []
        for _ in range(config['n_layers_temporal']):
            layer_list.append(TemporalLayer(config))

        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        # Sequentially apply each layer to the data
        for layer in self.layers:
                x = layer(x)
        return x


