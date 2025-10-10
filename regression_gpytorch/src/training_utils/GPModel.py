# Create the GP model
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import os
import sys
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from path_kernel import PathKernel
from group_kernel import GroupKernel
from matern_kernel_with_fix_scale import MaternKernelWithNormalizingScale, KeopsMaternKernelWithNormalizingScale

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, source_normalizer = 1.0,
                 site_normalizer = 1.0, path_normalizer = 1.0,
                 between_kernel = True, source_effect = True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True,
                                                  jitter_val = 2e-3
                                                  )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernels = []
        # Source effect
        # base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[1, 2, 3])
        # base_kernel._set_lengthscale(source_normalizer)
        if source_effect:
            base_kernel = MaternKernelWithNormalizingScale(
                nu = 1.5, active_dims = [4, 5, 6],
                normalizing_scale = source_normalizer
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.source_var_training = []
            self.source_len_training = []
            self.kernels.append('source')
        else:
            self.covar_module = None
        # Site effect
        # base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[4,5,6])
        # base_kernel._set_lengthscale(site_normalizer)
        base_kernel = MaternKernelWithNormalizingScale(
            nu = 1.5, active_dims = [7, 8, 9],
            normalizing_scale = site_normalizer
        )
        if self.covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        else:
            self.covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        self.site_var_training = []
        self.site_len_training = []
        self.kernels.append('site')
        # Path effect
        base_kernel = PathKernel(nu = 1.5, 
                       num_int_pts=5,
                       active_dims=[1,2,3,7,8,9],
                       normalizing_scale = path_normalizer,
                       normalize=False)
        # base_kernel._set_lengthscale(path_normalizer)
        self.covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        self.path_var_training = []
        self.path_len_training = []
        self.kernels.append('path')
        # Between event 
        # Add a constraint to the variance of between event variance
        # eta_constraint = gpytorch.constraints.constraints.Interval(0.05, 0.5)
        # self.covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0), outputscale_constraint = eta_constraint)
        if between_kernel:
            self.covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0))
            self.eta_training = []
            self.kernels.append('eta')

        self.likelihood_noise = []

        # Regeister kernels as a buffered parameter for saving and loading
        for kernel_name in self.kernels:
            var_name = f"{kernel_name}_kernel"
            var_value = torch.tensor(self.kernels.index(kernel_name))
            self.register_buffer(var_name, var_value)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 source_normalizer = 1.0, site_normalizer = 1.0, path_normalizer = 1.0,
                 between_kernel = True,
                 source_effect = True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernels = []
        # Source effect
        if source_effect:
            base_kernel = MaternKernelWithNormalizingScale(
                nu = 1.5, active_dims = [4, 5, 6],
                normalizing_scale = source_normalizer
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.source_var_training = []
            self.source_len_training = []
            self.kernels.append('source')
        else:
            self.covar_module = None
        # Site effect
        base_kernel = MaternKernelWithNormalizingScale(
            nu = 1.5, active_dims = [7, 8, 9],
            normalizing_scale = site_normalizer
        )
        if self.covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        else:
            self.covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        self.site_var_training = []
        self.site_len_training = []
        self.kernels.append('site')
        # Path effect
        base_kernel = PathKernel(nu = 1.5, 
                       num_int_pts=5,
                       active_dims=[1,2,3,7,8,9],
                       normalizing_scale = path_normalizer,
                       normalize=False)
        self.covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        self.path_var_training = []
        self.path_len_training = []
        self.kernels.append('path')
        # Between event 
        # Add a constraint to the variance of between event variance
        # eta_constraint = gpytorch.constraints.constraints.Interval(0.05, 0.5)
        # self.covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0), outputscale_constraint = eta_constraint)
        if between_kernel:
            self.covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0))
            self.eta_training = []
            self.kernels.append('eta')

        self.likelihood_noise = []

        # Regeister kernels as a buffered parameter for saving and loading
        for kernel_name in self.kernels:
            var_name = f"{kernel_name}_kernel"
            var_value = torch.tensor(self.kernels.index(kernel_name))
            self.register_buffer(var_name, var_value)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class MultiDeviceExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices, output_device,
                 source_normalizer = 1.0, site_normalizer = 1.0, path_normalizer = 1.0):
        super(MultiDeviceExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernels = []
        # Source effect
        base_kernel = KeopsMaternKernelWithNormalizingScale(
            nu = 1.5, active_dims = [4, 5, 6],
            normalizing_scale = source_normalizer
        )
        base_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.source_var_training = []
        self.source_len_training = []
        self.kernels.append('source')
        # Site effect
        base_kernel = KeopsMaternKernelWithNormalizingScale(
            nu = 1.5, active_dims = [7, 8, 9],
            normalizing_scale = site_normalizer
        )
        base_covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        self.site_var_training = []
        self.site_len_training = []
        self.kernels.append('site')
        # Path effect
        # base_kernel = PathKernel(nu = 1.5, 
        #                num_int_pts=5,
        #                active_dims=[1,2,3,7,8,9],
        #                normalizing_scale = path_normalizer,
        #                normalize=False)
        # base_covar_module += gpytorch.kernels.ScaleKernel(base_kernel)
        # self.path_var_training = []
        # self.path_len_training = []
        # self.kernels.append('path')
        # Between event 
        # Add a constraint to the variance of between event variance
        # eta_constraint = gpytorch.constraints.constraints.Interval(0.05, 0.5)
        # self.covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0), outputscale_constraint = eta_constraint)
        base_covar_module += gpytorch.kernels.ScaleKernel(GroupKernel(active_dims=0))
        self.eta_training = []
        self.kernels.append('eta')

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

        self.likelihood_noise = []
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)