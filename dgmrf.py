import torch
from layers.flex import FlexLayer

class DGMRF(torch.nn.Module):
    def __init__(self, graph, config):
        super(DGMRF, self).__init__()

        self.n_time = config["n_time"]

        layer_list = []
        for _ in range(config["n_layers"]):
            layer_list.append(FlexLayer(graph, config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, data, transpose=False, with_bias=True):
        x = data.x
        edge_index = data.edge_index

        if transpose:
            # Transpose operation means reverse layer order
            layer_iter = reversed(self.layers)
        else:
            layer_iter = self.layers

        for layer in layer_iter:
            x = layer(x, edge_index, transpose, with_bias)

        return x

    def log_det(self):
        # Sum log-determinants of all layers
        return sum([layer.log_det() for layer in self.layers])
    