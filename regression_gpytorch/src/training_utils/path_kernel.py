import gpytorch
import torch
from numpy.polynomial.legendre import leggauss
import math
from gpytorch.constraints import Positive
from typing import Optional

class PathKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def __init__(self, nu = 1.5, num_int_points=5, 
                 normalizing_scale: Optional[float] = 1.0, 
                #  length_prior=None, length_constraint=None, 
                 normalize=False, **kwargs):
        super(PathKernel, self).__init__(**kwargs)
        self.num_int_points = num_int_points
        # self.normalizing_scale = torch.nn.Parameter(torch.tensor(normalizing_scale))
        # Precompute Legendre-Gauss nodes and weights
        self.nodes, self.weights = leggauss(num_int_points)
        # Move nodes from [-1, 1] to [0, 1]
        self.nodes = 0.5 * (self.nodes + 1)
        self.weights = 0.5 * self.weights
        self.nu = nu
        self.normalize = normalize
        if normalizing_scale is not None and not torch.is_tensor(normalizing_scale):
            normalizing_scale = torch.tensor(normalizing_scale, dtype=torch.float32)
        self.register_buffer(
            'normalizing_scale', 
            normalizing_scale if normalizing_scale is not None else torch.tensor(1.0)
        )

    def path_dist(self, x1, x2, **params):
        # x1 = x1 * self.length
        # x2 = x2 * self.length
        x1_src = x1[:, 0:3]
        x1_sit = x1[:, 3:6]
        x2_src = x2[:, 0:3]
        x2_sit = x2[:, 3:6]
        buffer = torch.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)
        for i in range(self.num_int_points):
            for j in range(self.num_int_points):
                x1_i = x1_src + (x1_sit - x1_src) * self.nodes[i]
                x2_i = x2_src + (x2_sit - x2_src) * self.nodes[j]
                dist_i = self.covar_dist(x1_i, x2_i, **params).div(self.lengthscale).div(self.normalizing_scale)

                # Inplace addition to buffer
                buffer.add_(self.weights[i] * self.weights[j] * dist_i)
        return buffer
    
    # this is the kernel function, this function is written by ChatGPT
    def forward(self, x1, x2, **params):
        # x1 = x1 * self.length
        # x2 = x2 * self.length
        x1_src = x1[:, 0:3]
        x1_sit = x1[:, 3:6]
        x2_src = x2[:, 0:3]
        x2_sit = x2[:, 3:6]
        buffer = torch.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)
        for i in range(self.num_int_points):
            for j in range(self.num_int_points):
                x1_i = x1_src + (x1_sit - x1_src) * self.nodes[i]
                x2_i = x2_src + (x2_sit - x2_src) * self.nodes[j]
                dist_i = self.covar_dist(x1_i, x2_i, **params).div(self.lengthscale).div(self.normalizing_scale)

                # Below are copied from the matern_kernel
                exp_component = torch.exp(-math.sqrt(self.nu * 2) * dist_i)
                if self.nu == 0.5:
                    constant_component = 1
                elif self.nu == 1.5:
                    constant_component = (math.sqrt(3) * dist_i).add(1)
                elif self.nu == 2.5:
                    constant_component = (math.sqrt(5) * dist_i).add(1).add(5.0 / 3.0 * dist_i**2)
                # Inplace addition to buffer
                buffer.add_(self.weights[i] * self.weights[j] * constant_component.mul(exp_component))
        if self.normalize:
            x1_length = torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)
            x2_length = torch.ones(x2.shape[0], dtype=x2.dtype, device=x2.device)
        else:
            x1_length = torch.norm(x1_sit - x1_src, dim=1, keepdim=False)
            x2_length = torch.norm(x2_sit - x2_src, dim=1, keepdim=False)
        buffer = buffer * torch.outer(x1_length, x2_length)
        if self.training and (x1.shape == x2.shape):
            diag = torch.diag(buffer)
            buffer.div_(torch.sqrt(torch.outer(diag, diag)))
        else:
            diag1 = self.compute_kernel_integral_t(x1) * x1_length
            diag2 = self.compute_kernel_integral_t(x2) * x2_length
            buffer.div_(torch.outer(diag1, diag2))
        # if self.normalize:
        #     if self.training and (x1.shape == x2.shape):
        #         diag = torch.diag(buffer)
        #         buffer.div_(torch.sqrt(torch.outer(diag,diag)))
        #     else:
        #         diag1 = self.compute_kernel_integral_t(x1)
        #         diag2 = self.compute_kernel_integral_t(x2)
        #         buffer.div_(torch.sqrt(torch.outer(diag1, diag2)))
        # else:
        #     x1_length = torch.norm(x1_sit - x1_src, dim=1, keepdim=False)
        #     x2_length = torch.norm(x2_sit - x2_src, dim=1, keepdim=False)
        #     buffer = buffer * torch.outer(x1_length, x2_length)
        return buffer
    
    def compute_kernel_integral_t(self, x):
        x1_src = x[:, 0:3]
        x1_sit = x[:, 3:6]
        x2_src = x[:, 0:3]
        x2_sit = x[:, 3:6]
        buffer = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for i in range(self.num_int_points):
            for j in range(self.num_int_points):
                x1_i = x1_src + (x1_sit - x1_src) * self.nodes[i]
                x2_i = x2_src + (x2_sit - x2_src) * self.nodes[j]
                dist_i = self.covar_dist(x1_i, x2_i, diag = True).div(self.lengthscale).div(self.normalizing_scale).flatten()
                # Below are copied from the matern_kernel
                exp_component = torch.exp(-math.sqrt(self.nu * 2) * dist_i)
                if self.nu == 0.5:
                    constant_component = 1
                elif self.nu == 1.5:
                    constant_component = (math.sqrt(3) * dist_i).add(1)
                elif self.nu == 2.5:
                    constant_component = (math.sqrt(5) * dist_i).add(1).add(5.0 / 3.0 * dist_i**2)
                # Inplace addition to buffer
                buffer.add_(self.weights[i] * self.weights[j] * constant_component.mul(exp_component))
        return buffer
    
    def compute_kernel_integral_t_cpu(self, x):
        x1_src = x[:, 0:3].cpu()
        x1_sit = x[:, 3:6].cpu()
        x2_src = x[:, 0:3].cpu()
        x2_sit = x[:, 3:6].cpu()
        buffer = torch.zeros(x.shape[0], dtype=x.dtype, device='cpu')
        for i in range(self.num_int_points):
            for j in range(self.num_int_points):
                x1_i = x1_src + (x1_sit - x1_src) * self.nodes[i]
                x2_i = x2_src + (x2_sit - x2_src) * self.nodes[j]
                dist_i = self.covar_dist(x1_i, x2_i, diag = True).div(self.lengthscale).div(self.normalizing_scale)

                # Below are copied from the matern_kernel
                exp_component = torch.exp(-math.sqrt(self.nu * 2) * dist_i)
                if self.nu == 0.5:
                    constant_component = 1
                elif self.nu == 1.5:
                    constant_component = (math.sqrt(3) * dist_i).add(1)
                elif self.nu == 2.5:
                    constant_component = (math.sqrt(5) * dist_i).add(1).add(5.0 / 3.0 * dist_i**2)
                # Inplace addition to buffer
                buffer.add_(self.weights[i] * self.weights[j] * constant_component.mul(exp_component))
        return buffer



