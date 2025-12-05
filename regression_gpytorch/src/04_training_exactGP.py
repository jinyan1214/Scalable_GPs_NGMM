import math
import shutil
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import h5py
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from training_utils.GPModel import ExactGPModel
from training_utils.utils import set_seed, read_hdf5, predict_dataset_pre_select
from training_utils.utils import str2bool
import time
from torch.utils.data import TensorDataset, DataLoader
import argparse


def plot_training_loss(training_loss, model, save_dir):
    fig, axes = plt.subplots(1,3,figsize=(16, 4))
    for label, color in zip(['source', 'site', 'path'], ['blue', 'red', 'green']):
        if label in model.kernels:
            axes[0].plot(getattr(model, f"{label}_len_training"), label=f"{label} length scale", c = color)
    for label, color in zip(['source', 'site', 'path'], ['blue', 'red', 'green']):
        if label in model.kernels:
            axes[1].plot(getattr(model, f"{label}_var_training"), '--', label=f"{label} variance", c = color)
    if 'eta' in model.kernels:
        axes[1].plot(model.eta_training, ':', label='Between event variance')

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Hyperparameter Value')
    axes[0].legend()
    axes[1].plot(model.likelihood_noise, ':', label='likelihood noise')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Hyperparameter Value')
    axes[1].legend()

    axes[2].set_ylim([0,10])
    axes[2].plot(training_loss, label = "LOO loss")
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Training loss')
    axes[2].legend()
    
    fig.savefig(os.path.join(save_dir, 'training_loss_parameters.png'), dpi=300)
    plt.close(fig)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format training data for regression with GPyTorch')
    parser.add_argument('--num_training_eqs', type=int, default=100,
                        help='Number of earthquakes to use for training')
    parser.add_argument('--num_rlzs', type=str, default='1',
                        help='Number of realizations to use for training, can be "all" or an integer')
    parser.add_argument('--train_on_mean', type=str2bool, default='False',
                        help='If train on mean of realizations')                                                
    parser.add_argument('--num_testing_eqs', type=int, default=4179,
                        help='Number of earthquakes to use for testing')
    parser.add_argument('--random_seed', type=int, default=62,
                        help='Seed for random permutation of earthquake IDs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs to train the model')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use for training')
    parser.add_argument('--between_kernel', type=str2bool, default='True',
                        help='If include between event kernel')
    parser.add_argument('--source_effect', type=str2bool, default='True',
                        help='If include source effect kernel')
    parser.add_argument('--overwrite', type=str2bool, default='False',
                        help='If overwrite existing formatted data')                        

    args = parser.parse_args()
    # Set random seed
    set_seed(args.random_seed)
    num_training_eqs = args.num_training_eqs
    num_rlzs = args.num_rlzs
    num_testing_eqs = args.num_testing_eqs
    num_epochs = args.num_epochs
    optimizer_name = args.optimizer
    between_kernel = args.between_kernel
    source_effect = args.source_effect
    overwrite = args.overwrite
    train_on_mean = args.train_on_mean

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(current_dir), "output")
    dir_name = f"{num_training_eqs}_training_eqs_{num_epochs}_epochs"
    if not source_effect:
        dir_name += '_no_source'
    if not between_kernel:
        dir_name += '_no_between'
    if train_on_mean:
        dir_name += '_train_on_mean'
    trained_model_dir = os.path.join(output_dir, 'trained_models_exact', dir_name, f"{num_rlzs}_rlzs")
    if os.path.exists(trained_model_dir):
        if overwrite:
            shutil.rmtree(trained_model_dir)
        else:
            print(f"Trained model directory {trained_model_dir} already exists. Use --overwrite True to overwrite.")
            raise RuntimeError()
    os.makedirs(trained_model_dir, exist_ok=True)

    # Load training data
    if train_on_mean:
        train_data_file = os.path.join(output_dir,'formatted_data', 
                                    f'training_{num_training_eqs}_eqs', 
                                    f'training_sites_training_eqs_2.00_ASK14_{num_training_eqs}_eqs_{num_rlzs}_var_per_eq_mean.h5')
    else:                                       
        train_data_file = os.path.join(output_dir,'formatted_data', 
                                    f'training_{num_training_eqs}_eqs', 
                                    f'training_sites_training_eqs_2.00_ASK14_{num_training_eqs}_eqs_{num_rlzs}_var_per_eq.h5')
    train_x, train_y = read_hdf5(train_data_file)
    print(f"Number of training samples: {train_x.shape[0]}")
    print(f"Number of training features: {train_x.shape[1]}")
    # Centeralize y:
    train_y_mean = np.mean(train_y)
    np.savetxt(os.path.join(trained_model_dir, 'train_y_mean.txt'), np.array([train_y_mean]), fmt='%.6f')
    train_y = train_y - train_y_mean
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training")
    else:
        device = torch.device("cpu")
        # raise RuntimeError("No GPU available for training. Please check your setup.")
    # train_x = train_x.to(device)
    # train_y = train_y.to(device)

    # Define some training parameters
    batch_size = min(10000, train_x.shape[0])
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")

    # These are kernel normalizers that are estimated by calculating the average
    # kernel distance from 100 training earthquakes
    source_normalizer = 92.8112
    site_normalizer = 40.3403
    path_normalizer = 76.3133
    print(f"Kernel lengthscale normalizing constants: source {source_normalizer}, site {site_normalizer}, path {path_normalizer}")

    # Create training dataset and dataloader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    # Create the GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPModel(
        torch.Tensor([]), torch.Tensor([]), likelihood,
                    source_normalizer=source_normalizer, 
                    site_normalizer=site_normalizer, 
                    path_normalizer=path_normalizer,
                    between_kernel=between_kernel,
                    source_effect=source_effect
                    )

    # Set initial scale for the effects based on observations in preliminary runs
    if 'path' in model.kernels:
        var_init = 0.1
        lengthscale = 0.4
        kernel_ind = model.kernels.index('path')
        model.covar_module.kernels[kernel_ind].outputscale = var_init
        model.covar_module.kernels[kernel_ind].base_kernel.lengthscale = lengthscale
        print(f"Path kernel initial variance is set to {var_init}, lengthscale is set to {lengthscale}")
    if 'source' in model.kernels:
        var_init = 0.1
        lengthscale = 0.4
        kernel_ind = model.kernels.index('source')
        model.covar_module.kernels[kernel_ind].outputscale = var_init
        model.covar_module.kernels[kernel_ind].base_kernel.lengthscale = lengthscale
        print(f"Source kernel initial variance is set to {var_init}, lengthscale is set to {lengthscale}")
    if 'site' in model.kernels:
        var_init = 0.1
        lengthscale = 0.4
        kernel_ind = model.kernels.index('site')
        model.covar_module.kernels[kernel_ind].outputscale = var_init
        model.covar_module.kernels[kernel_ind].base_kernel.lengthscale = lengthscale
        print(f"Site kernel initial variance is set to {var_init}, lengthscale is set to {lengthscale}")
    if 'eta' in model.kernels:
        kernel_ind = model.kernels.index('eta')
        model.covar_module.kernels[kernel_ind].outputscale = 0.1
    likelihood.noise = 0.1

    
    # Use the adam optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}
        ], lr=0.01)  # Includes GaussianLikelihood parameters
    elif optimizer_name == 'lbfgs':
        # optimizer = FullBatchLBFGS([
        #     {'params': model.parameters()}
        # ], lr=0.01)  # Includes GaussianLikelihood parameters
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=20, 
                                      history_size=10, line_search_fn='strong_wolfe')
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Use the LeaveOneOutPseudoLikelihood loss
    mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
    training_loss = []

    # Initialize model saving directory and save cpu state
    training_loss_file = os.path.join(trained_model_dir, 'training_hyperparameter_loss.npy')
    model_save_path_pre = os.path.join(trained_model_dir, 'cpu_model_initial.pth')
    model_state = model.state_dict()
    torch.save(model_state, model_save_path_pre)
    likelihood_save_path_pre = os.path.join(trained_model_dir, 'cpu_likelihood_initial.pth')
    likelihood_state = likelihood.state_dict()
    torch.save(likelihood_state, likelihood_save_path_pre)

    # Move to GPU
    import subprocess

    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"]
    )
    print(output.decode().strip())

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cuda_mem_allo = torch.cuda.memory_allocated() / 1e9  # GB actively used
    cuda_mem_reserved = torch.cuda.memory_reserved() / 1e9   # GB cached
    free, total = torch.cuda.mem_get_info()
    print(f"Before moving to GPU: cuda_alloc={cuda_mem_allo:.2f} GB, cuda_reserved={cuda_mem_reserved:.2f} GB")
    print(f"GPU memory free: {free/1e9:.2f} GB, total: {total/1e9:.2f} GB")
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        # Save the gpu model
        model_save_path_pre = os.path.join(trained_model_dir, 'gpu_model_initial.pth')
        model_state = model.state_dict()
        torch.save(model_state, model_save_path_pre)
        likelihood_save_path_pre = os.path.join(trained_model_dir, 'gpu_likelihood_initial.pth')
        likelihood_state = likelihood.state_dict()
        torch.save(likelihood_state, likelihood_save_path_pre)

        model_save_path_post = os.path.join(trained_model_dir, 'gpu_model_trained.pth')
        likelihood_save_path_post = os.path.join(trained_model_dir, 'gpu_likelihood_trained.pth')
    else:
        model_save_path_post = os.path.join(trained_model_dir, 'cpu_model_trained.pth')
        likelihood_save_path_post = os.path.join(trained_model_dir, 'cpu_likelihood_trained.pth')

    # Training loop
    model.train()
    likelihood.train()
    smallest_loss = np.inf
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        # minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
        # for x_batch, y_batch in minibatch_iter:
        for batch_i, (x_batch, y_batch) in enumerate(train_loader):
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            model.set_train_data(x_batch, y_batch, strict=False)
            
            optimizer.zero_grad()
            try:
                output = model(x_batch)
            except Exception as e:
                print(f"Error occurred while forwarding model: {e}")
                continue
            loss = -mll(output, y_batch)
            # minibatch_iter.set_postfix(loss=f"{loss.item():.4f}",
            #                           noise=f"{likelihood.noise.item():.4f}")
            cuda_mem_allo = torch.cuda.memory_allocated() / 1e9  # GB actively used
            cuda_mem_reserved = torch.cuda.memory_reserved() / 1e9   # GB cached

            epochs_iter.set_postfix(minibatch=batch_i, loss=f"{loss.item():.4f}",
                                    noise=f"{likelihood.noise.item():.4f}",
                                    cuda_alloc=f"{cuda_mem_allo:.2f} GB",
                                    cuda_reserved=f"{cuda_mem_reserved:.2f} GB",
                                    # source_lengthScale=f"{model.source_len_training[-1]*source_normalizer:.4f}" if len(model.source_len_training)>0 else 'NA'
                                    )
            loss.backward(retain_graph=True)
            # source event
            if 'source' in model.kernels:
                kernel_idx = model.kernels.index('source')
                model.source_len_training.append(model.covar_module.kernels[kernel_idx].base_kernel.lengthscale.item())
                model.source_var_training.append(model.covar_module.kernels[kernel_idx].outputscale.item())
            # site event
            if 'site' in model.kernels:
                kernel_idx = model.kernels.index('site')
                model.site_len_training.append(model.covar_module.kernels[kernel_idx].base_kernel.lengthscale.item())
                model.site_var_training.append(model.covar_module.kernels[kernel_idx].outputscale.item())
            # path event
            if 'path' in model.kernels:
                kernel_idx = model.kernels.index('path')
                model.path_len_training.append(model.covar_module.kernels[kernel_idx].base_kernel.lengthscale.item())
                model.path_var_training.append(model.covar_module.kernels[kernel_idx].outputscale.item())
            # within event
            if 'eta' in model.kernels:
                kernel_idx = model.kernels.index('eta')
                model.eta_training.append(model.covar_module.kernels[kernel_idx].outputscale.item())
            model.likelihood_noise.append(likelihood.noise.item())

            training_loss.append(loss.item())

            if loss.item() <= smallest_loss:
                smallest_loss = loss.item()
                torch.save(model.state_dict(), model_save_path_post)
                torch.save(likelihood.state_dict(), likelihood_save_path_post)
                best_epoch = i
            
            optimizer.step()

            # free GPU memory for x_batch and y_batch
            model.set_train_data(torch.Tensor([]).to(device), torch.Tensor([]).to(device), strict=False)
            del x_batch, y_batch, loss
            torch.cuda.empty_cache()

    print("Lowest loss achieved at epoch:", best_epoch)
    loss_array = np.array([
            model.site_var_training,
            model.site_len_training,
            model.path_var_training,
            model.path_len_training,
            model.likelihood_noise,
            training_loss]).T
    if source_effect:
        loss_array = np.hstack((
            np.array([model.source_var_training,
                      model.source_len_training]).T,
            loss_array))
    if between_kernel:
        loss_array = np.hstack((
            loss_array,
            np.array([model.eta_training]).T
            ))

    np.save(training_loss_file, loss_array)

    # Plot the evolution of the training loss and hyperparameters
    plot_training_loss(training_loss, model, trained_model_dir)

    # Load the model with lowest loss
    model.load_state_dict(torch.load(model_save_path_post))
    likelihood.load_state_dict(torch.load(likelihood_save_path_post))

    # Print the hyperparameters
    for param_name, raw_param in model.named_parameters():
        if param_name in ['variational_strategy.inducing_points', 
                        'variational_strategy._variational_distribution.variational_mean',
                        'variational_strategy._variational_distribution.chol_variational_covar']:
            continue
        constraint = model.constraint_for_parameter_name(f"{param_name}")
        if constraint is not None:
            value = constraint.transform(raw_param)
        else:
            value = raw_param
        print(f'Parameter: {param_name:50} transformed value = {value.item():.4f}, raw value = {raw_param.item():.4f}')

    if 'source' in model.kernels:
        source_index = model.kernels.index('source')
        source_scale = model.covar_module.kernels[source_index].base_kernel.lengthscale * source_normalizer
        source_scale = source_scale.item()
        source_variance = (model.covar_module.kernels[source_index].outputscale).item()
    else:
        source_scale = "NA"
        source_variance = "NA"
    if 'site' in model.kernels:
        site_index = model.kernels.index('site')
        site_scale = model.covar_module.kernels[site_index].base_kernel.lengthscale * site_normalizer
        site_scale = site_scale.item()
        site_variance = (model.covar_module.kernels[site_index].outputscale).item()
    else:
        site_scale = "NA"
        site_variance = "NA"
    if 'path' in model.kernels:
        path_index = model.kernels.index('path')
        path_scale = model.covar_module.kernels[path_index].base_kernel.lengthscale * path_normalizer
        path_scale = path_scale.item()
        path_variance = (model.covar_module.kernels[path_index].outputscale).item()
    else:
        path_scale = "NA"
        path_variance = "NA"
    if 'eta' in model.kernels:
        eta_variance = (model.covar_module.kernels[model.kernels.index('eta')].outputscale).item()
    else:
        eta_variance = "NA"
    print('Lengthscale:')
    print('Source:', source_scale, 'Site:', site_scale, 'Path:', path_scale)
    print('Variance:')
    print('Source:', source_variance,
          'Site:', site_variance,
          'Path:', path_variance)
    print('Between event variance:', eta_variance)
    print('Likelihood noise:', likelihood.noise.item())

    # Find the model's performance
    # model.eval()
    # likelihood.eval()

    # site_conditional_limit = 2 * site_scale
    # source_conditional_limit = 2 * source_scale

    # # Get and predict the test data
    # test_data_file = os.path.join(output_dir,'formatted_data', 
    #                                f'testing_{num_testing_eqs}_eqs', 
    #                                f'training_sites_testing_eqs_2.00_ASK14_{num_testing_eqs}_eqs_1_var_per_eq.h5')
    # test_x, test_y = read_hdf5(test_data_file)
    # print(f"Number of testing samples: {test_x.shape[0]}")
    # print(f"Number of testing features: {test_x.shape[1]}")
    # # Centeralize y:
    # test_y = test_y - train_y_mean
    # with torch.no_grad():
    #     test_x = torch.from_numpy(test_x).to(device)
    #     test_y = torch.from_numpy(test_y).to(device)
    #     test_mean, test_var = predict_dataset_pre_select(
    #         test_x, batch_size, model, train_x, train_y, likelihood,
    #         contiguous = False,
    #         max_distance_site=site_conditional_limit, max_distance_source=source_conditional_limit
    #     )
    
    # # Predict on the training set
    # with torch.no_grad():
    #     train_mean, train_var = predict_dataset_pre_select(
    #         train_x, batch_size, model, train_x, train_y, likelihood,
    #         contiguous = False,
    #         max_distance_site=site_conditional_limit, max_distance_source=source_conditional_limit
    #     )

    # # Predict on the training earthquake test sites
    # # # Get the training earthquake test sites data
    # test_sites_train_eqs_file = os.path.join(output_dir, 'formatted_data',
    #                                          f'training_{num_training_eqs}_eqs',
    #                                          f'testing_sites_training_eqs_2.00_ASK14_{num_training_eqs}_eqs_1_var_per_eq.h5')
    # test_sites_train_x, test_sites_train_y = read_hdf5(test_sites_train_eqs_file)
    # test_sites_train_x, test_sites_train_y = test_sites_train_x.to(device), test_sites_train_y.to(device)
    # print(f"Number of training earthquake test sites: {test_sites_train_x.shape[0]}")
    # print(f"Number of training earthquake test features: {test_sites_train_x.shape[1]}")
    # # Centeralize y:
    # test_sites_train_y = test_sites_train_y - train_y_mean
    # with torch.no_grad():
    #     test_sites_train_x = torch.from_numpy(test_sites_train_x).to(device)
    #     test_sites_train_y = torch.from_numpy(test_sites_train_y).to(device)
    #     test_sites_train_mean, test_sites_train_var = predict_dataset_pre_select(
    #         test_sites_train_x, batch_size, model, train_x, train_y, likelihood,
    #         contiguous = False,
    #         max_distance_site=site_conditional_limit, max_distance_source=source_conditional_limit
    #     )

    # # Predict on the testing earthquake test sites
    # start = time.time()
    # # # Get the testing earthquake test sites data
    # test_sites_test_eqs_file = os.path.join(output_dir, 'formatted_data',
    #                                          f'testing_{num_testing_eqs}_eqs',
    #                                          f'testing_sites_testing_eqs_2.00_ASK14_{num_testing_eqs}_eqs_1_var_per_eq.h5')
    # test_sites_test_x, test_sites_test_y = read_hdf5(test_sites_test_eqs_file)
    # test_sites_test_x, test_sites_test_y = test_sites_test_x.to(device), test_sites_test_y.to(device)
    # print(f"Number of testing earthquake test sites: {test_sites_test_x.shape[0]}")
    # print(f"Number of testing earthquake test features: {test_sites_test_x.shape[1]}")
    # # Centeralize y:
    # test_sites_test_y = test_sites_test_y - train_y_mean
    # with torch.no_grad():
    #     test_sites_test_x = torch.from_numpy(test_sites_test_x).to(device)
    #     test_sites_test_y = torch.from_numpy(test_sites_test_y).to(device)
    #     test_sites_test_mean, test_sites_test_var = predict_dataset_pre_select(
    #         test_sites_test_x, batch_size, model, train_x, train_y, likelihood,
    #         contiguous = False,
    #         max_distance_site=site_conditional_limit, max_distance_source=source_conditional_limit
    #     )

    # if torch.cuda.is_available():
    #     train_x = train_x.cpu()
    #     train_y = train_y.cpu()
    #     test_x = test_x.cpu()
    #     test_y = test_y.cpu()
    #     test_sites_train_x = test_sites_train_x.cpu()
    #     test_sites_train_y = test_sites_train_y.cpu()
    #     test_sites_test_x = test_sites_test_x.cpu()
    #     test_sites_test_y = test_sites_test_y.cpu()
    #     torch.cuda.empty_cache()   

    # RMSE_test = np.sqrt(np.mean((test_y.numpy() - test_mean.numpy())**2))
    # print(f"Training sites Testing earthquakes RMSE error {RMSE_test:.3f}; Data mean: {test_y.numpy().mean():.3f}; Data Std: {np.std(test_y.numpy()):.3f}")

    # RMSE_train = np.sqrt(np.mean((train_y.numpy() - train_mean.numpy())**2))
    # print(f"Training sites Training earthquakes RMSE error {RMSE_train:.3f}; Data mean: {train_y.numpy().mean():.3f}; Data Std: {np.std(train_y.numpy()):.3f}")

    # RMSE_new_sites_train = np.sqrt(np.mean((test_sites_train_y.numpy() - 
    #                                         test_sites_train_mean.numpy())**2))
    # print(f"Testing Sites Training earthquakes RMSE error {RMSE_new_sites_train:.3f}; Data mean: {test_sites_train_y.numpy().mean():.3f}; Data Std: {np.std(test_sites_train_y.numpy()):.3f}")

    # RMSE_new_sites_test = np.sqrt(np.mean((test_sites_test_y.numpy() - test_sites_test_mean.numpy())**2))
    # print(f"Testing Sites Testing earthquakes RMSE error {RMSE_new_sites_test:.3f}; Data mean: {test_sites_test_y.numpy().mean():.3f}; Data Std: {np.std(test_sites_test_y.numpy()):.3f}")

    # # Make a plot of the prediction for the training and testing data
    # with torch.no_grad():
    #     # Initialize plot
    #     f, ax = plt.subplots(1, 1, figsize=(8, 6))

    #     # Plot training data as black stars
    #     ax.plot(train_x[:,0].numpy(), train_y.numpy(), 'k*')
    #     # Plot predictive means as blue line
    #     ax.plot(test_x[:,0].numpy(), test_y.numpy(), 'r.')
    #     ax.plot(test_x[:,0].numpy(), test_mean.numpy(), 'b.')
    #     # Shade between the lower and upper confidence bounds
    #     # ax.fill_between(test_x[:,0].numpy(), lower[:,0].numpy(), upper[:,0].numpy(), alpha=0.5)
    #     ax.set_ylim([-3, 3])
    #     ax.set_xlabel("Eq_var_id")
    #     ax.set_ylabel("Res SA 2.0s")
    #     ax.legend(['Training Data', 'Test Data', 'Test Prediction (mean)'])
    #     f.savefig(os.path.join(trained_model_dir, 'test_prediction.png'), dpi=300)
    #     plt.close(f)
    #     f, ax = plt.subplots(1, 1, figsize=(4, 4))
    #     ax.plot(test_mean.numpy(), test_y.numpy(), 'b.')
    #     ax.plot([-3, 3], [-3, 3], 'k--')
    #     ax.set_xlim([-3, 3])
    #     ax.set_ylim([-3, 3])
    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("Observation")

    model = model.cpu()
    likelihood = likelihood.cpu()
    torch.save(model.state_dict(), os.path.join(trained_model_dir, 'cpu_model_trained.pth'))
    torch.save(likelihood.state_dict(), os.path.join(trained_model_dir, 'cpu_likelihood_trained.pth'))