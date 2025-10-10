import gpytorch
from typing import Optional
import torch

class MaternKernelWithNormalizingScale(gpytorch.kernels.MaternKernel):
    is_stationary = True
    has_lengthscale = True

    def __init__(self, normalizing_scale: Optional[float] = 1.0, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernelWithNormalizingScale, self).__init__(**kwargs)
        self.nu = nu
        if normalizing_scale is not None and not torch.is_tensor(normalizing_scale):
            normalizing_scale = torch.tensor(normalizing_scale, dtype=torch.float32)
        self.register_buffer(
            'normalizing_scale', 
            normalizing_scale if normalizing_scale is not None else torch.tensor(1.0)
        )

    def forward(self, x1, x2, **params):
        # Scale the inputs by the normalizing scale
        x1_scaled = x1.div(self.normalizing_scale)
        x2_scaled = x2.div(self.normalizing_scale)

        # Call the parent class's forward method with scaled inputs
        return super(MaternKernelWithNormalizingScale, self).forward(x1_scaled, x2_scaled, **params)

class KeopsMaternKernelWithNormalizingScale(gpytorch.kernels.keops.MaternKernel):
    is_stationary = True
    has_lengthscale = True

    def __init__(self, normalizing_scale: Optional[float] = 1.0, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(KeopsMaternKernelWithNormalizingScale, self).__init__(**kwargs)
        self.nu = nu
        if normalizing_scale is not None and not torch.is_tensor(normalizing_scale):
            normalizing_scale = torch.tensor(normalizing_scale, dtype=torch.float32)
        self.register_buffer(
            'normalizing_scale', 
            normalizing_scale if normalizing_scale is not None else torch.tensor(1.0)
        )

    def forward(self, x1, x2, **params):
        # Scale the inputs by the normalizing scale
        x1_scaled = x1.div(self.normalizing_scale)
        x2_scaled = x2.div(self.normalizing_scale)

        # Call the parent class's forward method with scaled inputs
        return super(KeopsMaternKernelWithNormalizingScale, self).forward(x1_scaled, x2_scaled, **params)