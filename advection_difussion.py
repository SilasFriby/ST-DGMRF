import numpy as np
from numpy.random import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import igraph as ig
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


## Initalize parameters

# Model parameters
D = 0.01  # Diffusion coefficient
v = np.array([-0.3, 0.3])  # Velocity vector
sigma_obs = 0.01

# Spatial
n_lattice = 30 

# Time
n_time = 20  # Number of timesteps



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
S_k = 10 * identity_matrix - A
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

w = 9  # Width of the mask
n_time_mask = 10  
t_start = 3  
t_end = t_start + n_time_mask 
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
for t in range(n_time):

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


## Save degree matrix and adjacency matrix at pytorch tensors

torch.save(torch.tensor(degree_matrix, dtype=torch.float), 'dataset/advection_diffusion/degree_matrix.pt')
torch.save(torch.tensor(A, dtype=torch.float), 'dataset/advection_diffusion/adjacency_matrix.pt')


#### True posterior mean

## Create Q as sparse matrix - diagonal block matrix with Q_k in every block

# Number of nodes
n_nodes_spatial = n_lattice**2
n_total = n_time * n_nodes_spatial

# Initialize lists for row indices, column indices, and values
row_indices = []
col_indices = []
values = []

for k in tqdm(range(n_time)):
    # Calculate the starting index for this block
    start_index = k * n_nodes_spatial
    
    for i in range(n_nodes_spatial):
        for j in range(n_nodes_spatial):
            # Calculate the global row and column indices
            row_idx = start_index + i
            col_idx = start_index + j
            
            # Append the indices and value
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            if (k == 0):
                values.append(Q_0[i, j])
            else:
                values.append(Q_k[i, j])  # Assuming Q_k is 2D

# Convert lists to numpy arrays
row_indices = np.array(row_indices)
col_indices = np.array(col_indices)
values = np.array(values)

# Create the CSR matrix
size = (n_time * n_nodes_spatial, n_time * n_nodes_spatial)
Q = csr_matrix((values, (row_indices, col_indices)), shape=size)



## Create F and F^T as sparse matrices
## F is a block lower bi-diagonal matrix with I in the diagonal and -F_k in the subdiagonal
## F^T is a block upper bi-diagonal matrix with I in the diagonal and -F_k^T in the superdiagonal

# Convert F_k and its transpose to LIL format for easier construction
F_k_lil = lil_matrix(F_k)
F_k_T_lil = lil_matrix(F_k.T)  # Ensure F_k_T uses corrected transpose data

# Initialize LIL matrices for F and F_T for easier construction
F_lil = lil_matrix((n_total, n_total))
F_T_lil = lil_matrix((n_total, n_total))

for k in tqdm(range(n_time)):
    row_start = k * n_nodes_spatial
    row_end = (k + 1) * n_nodes_spatial
    
    # Identity blocks for F and F^T
    F_lil[row_start:row_end, row_start:row_end] = np.eye(n_nodes_spatial)
    F_T_lil[row_start:row_end, row_start:row_end] = np.eye(n_nodes_spatial)
    
    # -F_k blocks for F and -F_k^T blocks for F^T
    if k < n_time - 1:
        F_lil[row_end:row_end + n_nodes_spatial, row_start:row_end] = -F_k_lil
        F_T_lil[row_start:row_end, row_end:row_end + n_nodes_spatial] = -F_k_T_lil

# Convert LIL matrices to CSR for efficient storage and operations
F = F_lil.tocsr()
F_T = F_T_lil.tocsr()



## Create mask - a vector with True for observed values and False for missing values

for t in range(n_time):
    mask_t = ~np.isnan(rho_matrix_mask[:, t]).flatten()
    if t == 0:
        mask = mask_t
    else:# stack in columns
        mask = np.vstack((mask, mask_t))

# Convert mask to tensor
mask = torch.tensor(mask.flatten(), dtype=torch.bool)

# Create noise term
mask_noise_term = (1./sigma_obs**2) * mask

# Create noise_term_matrix as a sparse diag matrix with noise_term as the diagonal
mask_noise_term_matrix = csr_matrix((mask_noise_term.flatten(), (np.arange(n_total), np.arange(n_total))), shape=(n_total, n_total))


## Create Omega = F^T @ Q @ F as a sparse matrix

Omega = F_T @ Q @ F 

## Omega plus - lhs side of the equation for posterior mean  

Omega_plus = Omega + mask_noise_term_matrix

## Compute rhs of equation for posterior mean = Omega * mu + mask_noise_term * y
## mu is zero in the example, hence rhs consists of only mask_noise_term * y

# Compute rhs second term mask_noise_term * y
for t in range(n_time):
    y_t = obs_matrix[:, t].flatten()
    mask_t = ~np.isnan(rho_matrix_mask[:, t]).flatten()
    mask_y_t = mask_t * y_t
    if t == 0:
        mask_y = mask_y_t
    else: # append
        mask_y = np.append(mask_y, mask_y_t)

## Solve the linear system Omega_plus * x = b for x, which represents the posterior mean in this context

true_posterior_mean = spsolve(Omega_plus, mask_y)

# Convert x_posterior_mean to tensor matrix with n_time columns
true_posterior_mean_matrix = true_posterior_mean.reshape(n_nodes_spatial, n_time, order='F')

# Save the posterior mean as a tensor
true_posterior_mean_tensor = torch.tensor(true_posterior_mean_matrix, dtype=torch.float)
torch.save(true_posterior_mean_tensor, 'dataset/advection_diffusion/post_mean_true.pt')


    

    


