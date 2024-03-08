import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

torch.tensor([1.2, 3]).device
torch.set_default_device('mps')  # current device is 0
torch.tensor([1.2, 3]).device