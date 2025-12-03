import torch
import numpy as np
import h5py
import os
import gpytorch
from tqdm import tqdm
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
## Set random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    """Sets random seeds for PyTorch, NumPy, and Python's random module."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def read_hdf5(file_path):
    """Reads data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    # add site_z column 
    X = np.column_stack((X, np.zeros((X.shape[0], 1))))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return X, Y

def read_hdf5_with_site_id(file_path, dtype = np.float32):
    """Reads data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        if 'site_id' not in f:
            raise KeyError(f"'site_id' dataset not found in {file_path}, rerun 02_format_training_data.py to include site_id")
        site_id = f['site_id'][:]
    # add site_z column
    X = np.column_stack((X, np.zeros((X.shape[0], 1))))
    X = X.astype(dtype)
    Y = Y.astype(dtype)
    site_id = site_id.astype(str)
    return X, Y, site_id

def read_hdf5_with_site_id_rrup(file_path, dtype = np.float32):
    """Reads data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        if 'site_id' not in f:
            raise KeyError(f"'site_id' dataset not found in {file_path}, rerun 02_format_training_data.py to include site_id")
        site_id = f['site_id'][:]
        if 'rrup' not in f:
            raise KeyError(f"'rrup' dataset not found in {file_path}, rerun 02_format_training_data.py to include rrup")
        rrup = f['rrup'][:]
    # add site_z column
    X = np.column_stack((X, np.zeros((X.shape[0], 1))))
    X = X.astype(dtype)
    Y = Y.astype(dtype)
    site_id = site_id.astype(str)
    return X, Y, site_id, rrup

def predict_dataset(loader, model, likelihood, contiguous = False):
    mean = torch.tensor([0.])
    lower = torch.tensor([0.])
    upper = torch.tensor([0.])
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, _ in tqdm(loader, total=len(loader), 
                               desc=f"Predicting {len(loader.dataset)} samples in batches:"):
            if contiguous:
                x_batch = x_batch.contiguous()            
            observed_pred = likelihood(model(x_batch))
            mean = torch.cat([mean, observed_pred.mean.cpu()])
            lower_i, upper_i = observed_pred.confidence_region()
            lower = torch.cat([lower, lower_i.cpu()])
            upper = torch.cat([upper, upper_i.cpu()])
            torch.cuda.empty_cache()
    mean = mean[1:]
    lower = lower[1:]
    upper = upper[1:]
    return mean, lower, upper

def predict_dataset_mean_var(loader, model, likelihood, contiguous = False,
                             print_progress = False):
    mean = torch.tensor([0.]).to(loader.dataset.tensors[0].device)
    var = torch.tensor([0.]).to(loader.dataset.tensors[0].device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if print_progress:
            iterator = tqdm(loader, total=len(loader), 
                               desc=f"Predicting {len(loader.dataset)} samples in batches:")
        else:
            iterator = loader
        for x_batch, _ in iterator:
            if contiguous:
                x_batch = x_batch.contiguous()            
            observed_pred = likelihood(model(x_batch))
            mean = torch.cat([mean, observed_pred.mean])
            var = torch.cat([var, observed_pred.lazy_covariance_matrix.diagonal()])
            torch.cuda.empty_cache()
    mean = mean[1:]
    var = var[1:]
    return mean, var

def get_dist(train, target):
    distances = torch.zeros(train.shape[0]).to(train.device)
    for i in range(train.shape[0]):
        train_i = train[i, :]
        dist_i = torch.norm(train_i - target, dim=1)
        min_dist_i = torch.min(dist_i)
        distances[i] = min_dist_i
    return distances

        
def select_conditional_data(train_x, target_x, max_distance_site, max_distance_source):
    # Select training data based on source distance to target data
    target_source = target_x[:, 4:7]
    target_source_unique = torch.unique(target_source, dim=0)
    dist_to_target_source = get_dist(train_x[:, 4:7], target_source_unique)
    source_selected_indices = torch.where(dist_to_target_source <= max_distance_source)[0]
    # Select training data based on site distance to target data
    source_selected_train_x = train_x[source_selected_indices, :]
    target_site = target_x[:, 7:10]
    target_site_unique = torch.unique(target_site, dim=0)
    dist_to_target_site = get_dist(source_selected_train_x[:, 7:10], target_site_unique)
    site_selected_indices = torch.where(dist_to_target_site <= max_distance_site)[0]
    selected_indices = source_selected_indices[site_selected_indices].cpu().numpy()
    return selected_indices

def predict_dataset_pre_select(test_x, batch_size, model, train_x, train_y, 
                            likelihood, contiguous = False,
                            max_distance_site=20.0, max_distance_source=50.0):
    test_mean = torch.zeros((test_x.shape[0],)).to(test_x.device)
    test_var = torch.zeros((test_x.shape[0],)).to(test_x.device)
    test_eq_ids = test_x[:, 0]//1000
    unique_eq_ids = torch.unique(test_eq_ids)
    for eq_id in tqdm(unique_eq_ids, total=len(unique_eq_ids), desc=f"Predicting {len(test_x)} samples by earthquake:"):
        eq_selected_indices = torch.where(test_eq_ids == eq_id)[0]
        eq_selected_test_x = test_x[eq_selected_indices, :]
        test_site_loc = eq_selected_test_x[:, 7:10]
        test_unique_site_loc = torch.unique(test_site_loc, dim=0)
        for i in tqdm(range(test_unique_site_loc.shape[0]), desc=f"Predicting earthquake {int(eq_id.item())} with {test_unique_site_loc.shape[0]} sites:", leave=False):
            site_loc = test_unique_site_loc[i, :].unsqueeze(0)
            site_selected_indices = torch.where((test_site_loc == site_loc).all(dim=1))[0]
            site_selected_test_x = eq_selected_test_x[site_selected_indices, :]
            loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(site_selected_test_x, torch.zeros((site_selected_test_x.shape[0], 1))),
                batch_size=batch_size,
                shuffle=False
            )
            conditional_indices = select_conditional_data(train_x, site_selected_test_x, max_distance_site, max_distance_source)
            if len(conditional_indices) == 0:
                print(f"No conditional training data found for site {site_loc.cpu().numpy()}, using all training data")
                conditional_indices = np.random.choice(train_x.shape[0], size=min(batch_size, train_x.shape[0]), replace=False)
            conditional_train_x = train_x[conditional_indices, :].contiguous()
            conditional_train_y = train_y[conditional_indices].contiguous()
            model.set_train_data(inputs=conditional_train_x, targets=conditional_train_y, strict=False)
            mean_i, var_i = predict_dataset_mean_var(loader, model, likelihood, 
                                                     contiguous, print_progress=False)
            test_mean[eq_selected_indices[site_selected_indices]] = mean_i
            test_var[eq_selected_indices[site_selected_indices]] = var_i
            torch.cuda.empty_cache()
    return test_mean, test_var