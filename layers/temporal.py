import torch

class TemporalLayer(torch.nn.Module):
    def __init__(self, config):
        super(TemporalLayer, self).__init__()
        self.n_lattice = config['n_lattice']
        self.n_samples = config['n_training_samples']
        
        # Learnable parameters
        self.d = torch.nn.Parameter(torch.sqrt(torch.tensor(0.01)))  # d^l, initialized to sqrt(0.01)
        self.v1 = torch.nn.Parameter(torch.tensor(-0.3))  # v1^l, initialized to -0.3
        self.v2 = torch.nn.Parameter(torch.tensor(0.3))  # v2^l, initialized to 0.3
        self.b_f = torch.nn.Parameter(torch.tensor(0.0))  # b_f^l, initialized to 0

    def forward(self, x):
        n_samples = self.n_samples

        # Create identity matrix with batch dimension
        identity_matrix = torch.eye(self.n_lattice**2, dtype=x.dtype).unsqueeze(0).repeat(n_samples, 1, 1)

        # Create the matrix M with learnable parameters d, v1, v2
        M = self.create_matrix_M()
        
        # Convert M to a batched matrix
        M = M.unsqueeze(0).repeat(n_samples, 1, 1) 

        # Transition matrix F
        F = identity_matrix + M 

        # Batch matrix multiplication
        result = torch.bmm(F, x) + self.b_f  # self.b_f should be adjusted to match the dimensions if necessary

        return result

    def get_neighbor_nodes(self, node):

        n_lattice = self.n_lattice

        # Safety check - node should be within the lattice
        if node < 0 or node >= n_lattice**2:
            raise ValueError("Node index out of bounds")
        
        # Right neighbor
        if (node + 1) % n_lattice > 0:
            neighbor_right = node + 1
        else:
            neighbor_right = node - (n_lattice - 1)

        # Left neighbor
        if (node - 1) % n_lattice < n_lattice - 1:
            neighbor_left = node - 1
        else:
            neighbor_left = node + (n_lattice - 1)

        # Up neighbor
        if node - n_lattice >= 0:
            neighbor_up = node - n_lattice
        else:
            neighbor_up = node + n_lattice * (n_lattice - 1) 

        # Down neighbor
        if node + n_lattice < n_lattice**2:
            neighbor_down = node + n_lattice
        else:
            neighbor_down = node - n_lattice * (n_lattice - 1)

        # Result dictionary
        neighbor_dict = {
            "right": neighbor_right,
            "left": neighbor_left,
            "up": neighbor_up,
            "down": neighbor_down
        }

        # Return dictionary
        return neighbor_dict

    def create_matrix_M(self):
        n_lattice = self.n_lattice
        D = self.d**2
        v = torch.stack([self.v1, self.v2])
        
        # Initialize the matrix M with zeros
        M = torch.zeros((n_lattice**2, n_lattice**2), dtype=self.d.dtype)

        # Diagonal elements
        # Set diagonal entries equal to -4D. D.item is used to convert the tensor to a scalar
        M = M + torch.diag(torch.full((n_lattice**2,), -4 * D.item()))

        # Off-diagonal elements
        for node in range(n_lattice**2):
            neighbor_dict = self.get_neighbor_nodes(node)
            
            for neighbor_name, neighbor_node in neighbor_dict.items():
                direction = {
                    "right": torch.tensor([1, 0], dtype=self.d.dtype),
                    "left": torch.tensor([-1, 0],  dtype=self.d.dtype),
                    "up": torch.tensor([0, 1], dtype=self.d.dtype),
                    "down": torch.tensor([0, -1], dtype=self.d.dtype)
                }[neighbor_name]

                # Calculate the dot product of the velocity vector and the direction vector
                M[node, neighbor_node] = D - 0.5 * torch.dot(direction, v)
        
        return M

