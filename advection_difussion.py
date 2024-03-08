import numpy as np
from numpy.random import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import igraph as ig
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Initialize parameters
D = 0.01  # Diffusion coefficient
v = np.array([-0.3, 0.3])  # Velocity vector
n_lattice = 30 # Number of lattice points
n_time = 20  # Number of timesteps
sigma_obs = 0.01


## Create function for M matrix based on advection-diffusion equation

# Neighbor function
def get_neighbor_nodes(node, n_lattice):

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

# M Function
def create_matrix_M(n_lattice, D, v):
    
    # Initialize the matrix M with zeros
    M = np.zeros((n_lattice**2, n_lattice**2))

    # Diagonal elements
    # Set diagonal entries equal to -4D
    diag_index = np.diag_indices(n_lattice**2)
    M[diag_index] = -4 * D

    # Off-diagonal elements
    # Loop over each node in the lattice
    for node in range(n_lattice**2):

        # Find the neighbors of the current node
        neighbor_dict = get_neighbor_nodes(node, n_lattice)

        # Loop over the neighbors
        for neighbor_node, neighbor_name in zip(neighbor_dict.values(), neighbor_dict.keys()):
            
            # Create unit direction vector based on the neighbor name
            if neighbor_name == "right":
                direction = np.array([1, 0])
            elif neighbor_name == "left":
                direction = np.array([-1, 0])
            elif neighbor_name == "up":
                direction = np.array([0, 1])
            elif neighbor_name == "down":
                direction = np.array([0, -1])

            # Calculate the dot product of the velocity vector and the direction vector
            M[node, neighbor_node] = D - 0.5 * np.dot(direction, v)
    
    return M

# Create the matrix M
M = create_matrix_M(n_lattice, D, v)


## Create transition matrix F based  on equation (31)

# Identity matrix
identity_matrix = np.eye(n_lattice**2)

# Taylor series expansion to define F_adv-diff 
F = identity_matrix + M + (1/2) * M.dot(M) + (1/6) * M.dot(M).dot(M)


## Simulate the process

# Initial quantity S_0, Q_0 and rho_0
lattice = ig.Graph.Lattice(dim=[n_lattice, n_lattice], circular=True)  
A = np.array(lattice.get_adjacency())
S_0 = 4 * identity_matrix - A
Q_0 = S_0.transpose().dot(S_0)


# Function to approximate the inverse of a matrix using SVD - neccesary for near-singular matrices
def approximate_inverse(X):
    U, S, VT = np.linalg.svd(X)
    # Reciprocal of S, with conditioning for near-zero singular values
    epsilon = 1e-10  # Threshold for considering singular values as zero
    S_inv = np.array([1/s if s > epsilon else 0 for s in S])
    Sigma_inv = np.diag(S_inv)
    X_inv_approx = VT.T @ Sigma_inv @ U.T
    return X_inv_approx


# Initial state - rho_0
rho_0 = multivariate_normal(np.zeros(n_lattice**2), approximate_inverse(Q_0))  # Initial state
rho_matrix = np.zeros((n_lattice**2, n_time))
rho_matrix[:, 0] = rho_0


# For the noise terms noise_rho, we use a time-invariantprecision matrix Q_k = S_k^T S_k where S_k = (10 * I - A)
S_k = 4 * identity_matrix - A
Q_k = S_k.transpose().dot(S_k)

# Sst time-invariant transition matrix F_k = F ^ 4 in order to perform four time steps at each iteration
F_k = F.dot(F).dot(F).dot(F)

# Iterate over timesteps with progress bar
for i in tqdm(range(n_time-1)):
    # Noise term
    noise_rho = multivariate_normal(np.zeros(n_lattice**2), approximate_inverse(Q_k)) 

    # Update rho using equation (32)   
    rho_prev = rho_matrix[:, i] 
    rho = F_k.dot(rho_prev) + noise_rho

    # Insert rho in rho_matrix
    rho_matrix[:, i+1] = rho


# ## Create rho plot

# # Create a figure to hold all subplots
# plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

# # Loop through each time step
# for sample_time in range(n_time):
#     # Reshape the column at timestep_index to a 2D array for plotting
#     plot_data = rho_matrix[:, sample_time].reshape((n_lattice, n_lattice))
    
#     # Create a subplot for the current time step
#     plt.subplot(4, 5, sample_time + 1)  # Arguments are (rows, columns, subplot index)
#     plt.imshow(plot_data)
#     plt.colorbar()
#     plt.title(f"Time Step: {sample_time}")  # Optional: add a title to each subplot

# plt.tight_layout()  # Adjust subplots to fit in the figure area
# plt.show()


## Create masked observations

# Function to apply the mask with missing data
def apply_mask(rho_matrix, t_start, t_end, w, n_lattice):
    
    mask_start = (n_lattice - w) // 4 # Start index for the mask (left corner)
    for t in range(t_start, t_end):
        for i in range(mask_start, mask_start + w):
            for j in range(mask_start, mask_start + w):
                # Convert 2D index to 1D index
                index = i * n_lattice + j
                rho_matrix[index, t] = np.nan  # Applying the mask (set to NaN for missing data)
            
    return rho_matrix



## Mask the simulations

w = 9  # Mask width
n_time_mask = 10  # Number of timesteps to mask
t_start = 3  # Example start time, you can choose any start time
t_end = t_start + n_time_mask  # End time after 10 consecutive timesteps
rho_matrix_mask = apply_mask(rho_matrix.copy(), t_start, t_end, w, n_lattice)


## Create observations by adding noise term to the masked simulations with sd sigma

noise_obs = np.random.normal(0, sigma_obs, rho_matrix_mask.shape)
obs_matrix = rho_matrix + noise_obs # mask is applied later in vi.py

# ## Create observation plot

# # Create a figure to hold all subplots
# plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

# # Loop through each time step
# for sample_time in range(n_time):
#     # Reshape the column at timestep_index to a 2D array for plotting
#     plot_data = obs_matrix[:, sample_time].reshape((n_lattice, n_lattice))
    
#     # Create a subplot for the current time step
#     plt.subplot(4, 5, sample_time + 1)  # Arguments are (rows, columns, subplot index)
#     plt.imshow(plot_data)
#     plt.colorbar()
#     plt.title(f"Time Step: {sample_time}")  # Optional: add a title to each subplot

# plt.tight_layout()  # Adjust subplots to fit in the figure area
# plt.show()

## Create graph object

# Degree matrix of the lattice 
degree_matrix = np.diag(lattice.degree())

# Extract the edges from the lattice
edge_list = [(edge.source, edge.target) for edge in lattice.es]

# Convert to a tensor
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index)

# Eigen values
eigen_values_inverse_D_A, _ = np.linalg.eig(approximate_inverse(degree_matrix) @ A)
 
# Positions
x_coord, y_coord = np.meshgrid(range(n_lattice), range(n_lattice)) # Create a meshgrid of coordinates
pos = np.stack((x_coord.flatten(), y_coord.flatten()), axis=1) # Stack the coordinates in the correct shape (n_lattice*n_lattice, 2)



# Create graph Data objects for each time step and save
sample_time = [3, 4]
for t in sample_time:

    # Mask - create vector mask which eqauls False if the value is nan and True if the value is not nan
    mask = ~np.isnan(rho_matrix_mask[:, t]).flatten()

    # Create graph data object
    graph_data = Data(
        x=torch.tensor(obs_matrix[:, t], dtype=torch.float).view(-1, 1),
        edge_index=edge_index,
        pos=torch.tensor(pos, dtype=torch.float),
        mask=torch.tensor(mask, dtype=torch.bool),
        eigvals = torch.tensor(eigen_values_inverse_D_A, dtype=torch.float)
    )
    
    # Save graph data object as pickle file
    torch.save(graph_data, f'dataset/advection_diffusion/graph_y_{t}.pt')
    

    


