import numpy as np
import torch

np_ones_vector = np.ones(shape=(2,3))
ones_tensor = torch.from_numpy(ndarray=np_ones_vector)
## Working with tensors
zeros_tensor = torch.zeros(size=(3,4))



## Matrix
matrix_a = torch.randn(2,3)
matrix_b = torch.randn(3,2)

matrix_c = torch.matmul(input=matrix_a, other=matrix_b)