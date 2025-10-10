import gpytorch
import torch
class GroupKernel(gpytorch.kernels.Kernel):
    # This is important under the hood
    is_stationary = True

    # this is the kernel function, this function is written by ChatGPT
    def forward(self, x1, x2, **params):
        # Broadcasting comparison: shape (I, J, D)
        # print("x1:", x1.shape, "x2:", x2.shape)
        equality = x1[:, None, :] == x2[None, :, :]
        match = torch.all(equality, dim=2)  # shape (I, J)
        return match.int()