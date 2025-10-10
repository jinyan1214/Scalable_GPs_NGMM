import shutil
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from training_utils.GPModel import GPModel, ExactGPModel
import time
from torch.utils.data import TensorDataset, DataLoader, Subset
import argparse
from training_utils.utils import set_seed, read_hdf5_with_site_id, predict_dataset
import geopandas as gpd


from joblib import Parallel, delayed
import multiprocessing
import time
import contextlib
import joblib
from tqdm import tqdm
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def split_list(lst, n):
    """
    Split a list `lst` into `n` smaller lists as evenly as possible.
    """
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx  # return the index as well

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format training data for regression with GPyTorch')
    parser.add_argument('--num_training_eqs', type=int, default=400,
                        help='Number of earthquakes to use for training')
    # parser.add_argument('--num_testing_eqs', type=int, default=1000,
    #                     help='Number of earthquakes to use for testing')
    parser.add_argument('--random_seed', type=int, default=62,
                        help='Seed for random permutation of earthquake IDs')
    parser.add_argument('--num_epochs', type=int, default=2500,
                        help='Number of epochs to train the model')
    parser.add_argument('--num_conditional_eqs', type=int, default=100,
                        help='Number of conditional earthquakes to use in Exact GP')
    parser.add_argument('--num_candidate_eqs', type=int, default=4179,
                        help='Number of candidate earthquakes to use in Exact GP')
    parser.add_argument('--prediction_batch_size', type=int, default=1024,
                        help='Number of earthquakes to predict in each batch')

    args = parser.parse_args()
    num_training_eqs = args.num_training_eqs
    num_epochs = args.num_epochs
    num_conditional_eqs = args.num_conditional_eqs
    num_candidate_eqs = args.num_candidate_eqs
    prediction_batch_size = args.prediction_batch_size

    print(f"Number of training earthquakes: {num_training_eqs}, Number of epochs: {num_epochs}, Number of conditional earthquakes: {num_conditional_eqs}, Number of candidate earthquakes: {num_candidate_eqs}, Prediction batch size: {prediction_batch_size}")

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for prediction")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction")
    # Set random seed
    set_seed(args.random_seed)

    # Load the formatted data for all eq
    output_dir = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output"
    site_metadata_file = os.path.join(output_dir, 'sites_train_test_split', 'site_metadata.csv')
    site_metadata_df = pd.read_csv(site_metadata_file)
    site_metadata_df['site_z'] = 0
    format_data_dir = os.path.join(output_dir, 'formatted_data')
    total_num_eqs = 8358
    all_data_dir = os.path.join(format_data_dir, f'training_{total_num_eqs}_eqs')

    all_data_file_train = os.path.join(all_data_dir, f'training_sites_training_eqs_2.00_ASK14_{total_num_eqs}_eqs_1_var_per_eq.h5')
    all_data_x_train, all_data_y_train, all_data_site_id_train = read_hdf5_with_site_id(all_data_file_train)

    all_data_file_test = os.path.join(all_data_dir, f'testing_sites_training_eqs_2.00_ASK14_{total_num_eqs}_eqs_1_var_per_eq.h5')
    all_data_x_test, all_data_y_test, all_data_site_id_test = read_hdf5_with_site_id(all_data_file_test)

    all_data_x = np.concatenate((all_data_x_train, all_data_x_test), axis=0)
    all_data_y = np.concatenate((all_data_y_train, all_data_y_test), axis=0)
    all_data_site_id = np.concatenate((all_data_site_id_train, all_data_site_id_test), axis=0)

    train_y_mean = all_data_y_train[:num_training_eqs].mean()
    all_data_y = all_data_y - train_y_mean
    
    print(f"Loaded total number of data points: {all_data_x.shape[0]}")
    # Load the earthquake IDs
    rup_var_sample_file = os.path.join(all_data_dir, 'rup_var_sample_training.csv')
    rup_var_df = pd.read_csv(rup_var_sample_file)
    rup_var_df['eq_var_id'] = rup_var_df.apply(lambda row: int(row['eq_id']*1000 + row['rup_var_id']), axis=1)
    candidate_eq_var_ids = rup_var_df['eq_var_id'][:num_candidate_eqs].values
    testing_eq_var_ids = rup_var_df['eq_var_id'][total_num_eqs//2:].values

    candidate_mask = np.isin(all_data_x[:,0], candidate_eq_var_ids)
    testing_mask = np.isin(all_data_x[:,0], testing_eq_var_ids)
    print(f"Number of candidate points: {np.sum(candidate_mask)}, Number of testing points: {np.sum(testing_mask)}")

    candidate_x = all_data_x[candidate_mask, :]
    candidate_y = all_data_y[candidate_mask]
    candidate_site_id = all_data_site_id[candidate_mask]
    testing_x = all_data_x[testing_mask, :]
    testing_y = all_data_y[testing_mask]
    testing_site_id = all_data_site_id[testing_mask]
    # Move to device
    # testing_x = torch.from_numpy(testing_x).to(device)
    # testing_y = torch.from_numpy(testing_y).to(device)

    # create the model
    source_normalizer = 92.8112
    site_normalizer = 40.3403
    path_normalizer = 76.3133
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    inducing_points = torch.zeros((2000,7)) # Placeholder for initializing the model
    svmodel = GPModel(inducing_points=inducing_points, 
                    source_normalizer=source_normalizer, 
                    site_normalizer=site_normalizer, 
                    path_normalizer=path_normalizer)
    svmodel = svmodel.to(device)
    likelihood = likelihood.to(device)

    # Load the trained model state
    trained_dir = os.path.join(output_dir, 'trained_models', f'{num_training_eqs}_training_eqs_{num_epochs}_epochs')
    if device == torch.device("cuda"):
        model_state_file = os.path.join(trained_dir, 'gpu_model_trained.pth')
        likelihood_state_file = os.path.join(trained_dir, 'gpu_likelihood_trained.pth')
    else:
        model_state_file = os.path.join(trained_dir, 'cpu_model_trained.pth')
        likelihood_state_file = os.path.join(trained_dir, 'cpu_likelihood_trained.pth')
    
    svmodel.load_state_dict(torch.load(model_state_file))
    likelihood.load_state_dict(torch.load(likelihood_state_file))
    print(f"Loaded trained model from {trained_dir}")

    source_lengthscale = source_normalizer * svmodel.covar_module.kernels[0].base_kernel.lengthscale
    source_lengthscale = source_lengthscale.item()
    source_corr_threshold = 3 * source_lengthscale
    print(f"Source lengthscale: {source_lengthscale}, 3*lengthscale: {source_corr_threshold}")
    site_lengthscale = site_normalizer * svmodel.covar_module.kernels[1].base_kernel.lengthscale
    site_lengthscale = site_lengthscale.item()
    site_corr_threshold = 3 * site_lengthscale
    print(f"Site lengthscale: {site_lengthscale}, 3*lengthscale: {site_corr_threshold}")

    # Select conditional points 
    testing_data_df = pd.DataFrame(testing_x,
                                columns=['eq_var_id','source_cst_x',
                                        'source_cst_y', 'source_cst_z',
                                        'site_x', 'site_y', 'site_z'])
    testing_data_df['cybershake_res'] = testing_y
    testing_data_df['site_id'] = testing_site_id
    pred_df = testing_data_df[['eq_var_id', 'site_id']].copy()
    # Loop through site ids to select conditional points
    unique_site_ids = np.unique(testing_site_id)
    # for site_id in tqdm(unique_site_ids, total =len(unique_site_ids),
    #                     desc="Loop through testing sites"):
    for site_id in unique_site_ids:
        testing_data_site = testing_data_df[
            testing_data_df['site_id'] == site_id]
        testing_x_site = testing_data_site.values[:,0:7].astype(np.float32)
        testing_y_site = testing_data_site.values[:,7].astype(np.float32)
        # Site location within ...
        site_loc = site_metadata_df[
            site_metadata_df['site_id'] == site_id][['site_x', 'site_y', 'site_z']].values[0,:]
        site_dist = np.linalg.norm(
            site_metadata_df[['site_x', 'site_y', 'site_z']].values - site_loc.reshape((1,3)), axis=1)
        conditional_site_ids = site_metadata_df['site_id'].values[site_dist < site_corr_threshold]

        site_mask = np.isin(candidate_site_id, conditional_site_ids)
        if len(conditional_site_ids) == 0:
            print(f"Warning: No candidate points found for site {site_id}. Using all candidate points.")
            continue
        conditional_x_site = candidate_x[site_mask]
        conditional_y_site = candidate_y[site_mask]
        conditional_source_mask = np.zeros((conditional_x_site.shape[0],), dtype=bool)
        # Source location within ...
        for row_i, row in testing_data_site.iterrows():
            row_source_loc = row[['source_cst_x', 'source_cst_y', 'source_cst_z']].values.astype(np.float32)
            source_dist = np.linalg.norm(
                conditional_x_site[:,1:4] - row_source_loc.reshape((1,3)), axis=1)
            conditional_source_mask_i = source_dist < (source_corr_threshold)
            if np.sum(conditional_source_mask_i) == 0:
                eq_var_id = row['eq_var_id']
                print(f"Warning: No conditional points found for site {site_id}, eq_var_id {eq_var_id}.")
            conditional_source_mask = np.logical_or(conditional_source_mask, conditional_source_mask_i)
        conditional_x = conditional_x_site[conditional_source_mask]
        conditional_y = conditional_y_site[conditional_source_mask]
        print(f"Site {site_id}: Number of conditional points selected: {conditional_x.shape[0]}.")
        conditional_x = torch.from_numpy(conditional_x).to(device)
        conditional_y = torch.from_numpy(conditional_y).to(device)
        # Create an exact GP model and copy the hyperparameters
        model = ExactGPModel(conditional_x, conditional_y, likelihood)
        model.covar_module.load_state_dict(svmodel.covar_module.state_dict())
        model.mean_module.load_state_dict(svmodel.mean_module.state_dict())
    
        # Evaluate the model on the rest of the data
        model = model.to(device)
        likelihood = likelihood.to(device)

        model.eval()
        likelihood.eval()
        print("Evaluating the Exact GP model on the test data...")

        test_x_site = torch.from_numpy(testing_x_site).to(device)
        test_y_site = torch.from_numpy(testing_y_site).to(device)

        test_dataset = DatasetWithIndex(TensorDataset(test_x_site, test_y_site))
        test_loader = DataLoader(test_dataset, batch_size = prediction_batch_size,
                                  shuffle=False)
        pred_mean_site = torch.zeros((len(test_dataset),))
        pred_var_site = torch.zeros((len(test_dataset),))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for x_batch, _, idx in tqdm(test_loader, total=len(test_loader), 
                                   desc=f"Predicting {len(test_loader.dataset)} samples in batches:"):
                observed_pred = model(x_batch)
                # pred_mean = torch.cat([pred_mean, observed_pred.mean.cpu()])
                # pred_var = torch.cat([pred_var, observed_pred.lazy_covariance_matrix.diagonal().cpu()])
                pred_mean_site[idx] = observed_pred.mean.cpu()
                pred_var_site[idx] = observed_pred.lazy_covariance_matrix.diagonal().cpu()
                torch.cuda.empty_cache()
        pred_mean_site = pred_mean_site.numpy()
        pred_var_site = pred_var_site.numpy()

        pred_df_site = pd.DataFrame({
            'eq_var_id': testing_data_site['eq_var_id'].values.astype(int),
            'pred_mean': pred_mean_site + train_y_mean,
            'pred_var': pred_var_site,
            'site_id': testing_data_site['site_id'].values
        })
        pred_df = pred_df.merge(
            pred_df_site, on=['eq_var_id', 'site_id'], how='left')



    # n_batches = int(np.ceil(testing_x.shape[0] / prediction_batch_size))
    # all_iter_list = list(range(len(testing_x)))
    # splited_iter_list = split_list(all_iter_list, n_batches)

    # def single_batch_predict(batch_i):
    #     global testing_x, model, splited_iter_list
    #     subset_indices = splited_iter_list[batch_i]
    #     print(len(subset_indices))
    #     x_batch = testing_x[subset_indices]
    #     observed_pred = model(x_batch)
    #     pred_mean = observed_pred.mean.cpu().numpy()
    #     pred_var = observed_pred.lazy_covariance_matrix.diagonal().cpu().numpy()
    #     return pred_mean, pred_var

    # ### Define the number of threads
    # num_cores = min(2, multiprocessing.cpu_count())
    # print(f"Using cores: {num_cores}")
    # print(f"Number of cores available: {multiprocessing.cpu_count()}")

    # with tqdm_joblib(tqdm(desc="Loop through training sites", total=n_batches)) as progress_bar:
    #     results = Parallel(n_jobs=num_cores)(delayed(single_batch_predict)(
    #         batch_i
    #     ) for batch_i in range(n_batches))

    # pred_mean = np.concatenate([result[0] for result in results], axis=0)
    # pred_var = np.concatenate([result[1] for result in results], axis=0)

    # test_dataset = DatasetWithIndex(TensorDataset(testing_x, testing_y))
    # test_loader = DataLoader(test_dataset, batch_size = prediction_batch_size,
    #                           shuffle=False)

    # # pred_mean = torch.tensor([0.])
    # # pred_var = torch.tensor([0.])

    # pred_mean = torch.zeros((len(test_dataset),))
    # pred_var = torch.zeros((len(test_dataset),))

    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     for x_batch, _, idx in tqdm(test_loader, total=len(test_loader), 
    #                            desc=f"Predicting {len(test_loader.dataset)} samples in batches:"):
    #         observed_pred = model(x_batch)
    #         # pred_mean = torch.cat([pred_mean, observed_pred.mean.cpu()])
    #         # pred_var = torch.cat([pred_var, observed_pred.lazy_covariance_matrix.diagonal().cpu()])
    #         pred_mean[idx] = observed_pred.mean.cpu()
    #         pred_var[idx] = observed_pred.lazy_covariance_matrix.diagonal().cpu()
    #         torch.cuda.empty_cache()
    # # pred_mean = pred_mean[1:].numpy()
    # # pred_var = pred_var[1:].numpy()
    # pred_mean = pred_mean.numpy()
    # pred_var = pred_var.numpy()

    # Save the predictions
    prediction_dir = os.path.join(output_dir, f'prediction_outputs_{device}', 
                                  f'{num_training_eqs}_training_eqs_{num_epochs}_epochs_{num_conditional_eqs}_condition_eqs')
    if os.path.exists(prediction_dir):
        shutil.rmtree(prediction_dir)
    os.makedirs(prediction_dir, exist_ok=True)

    # Use geospatial merge to get the site_id (This does not work because multiple site_id can have the same location)
    # test_data_df = pd.DataFrame(testing_x.cpu().numpy(),
    #                             columns=['eq_var_id','source_cst_x',
    #                                      'source_cst_y', 'source_cst_z',
    #                                      'site_x', 'site_y', 'site_z'])
    # test_data_gdf = gpd.GeoDataFrame(
    #     test_data_df,
    #     geometry=gpd.points_from_xy(test_data_df['site_x']*1000, test_data_df['site_y']*1000),
    #     crs="EPSG:32611" # As specified in the metadata
    # )
    # # Load the site_id metadata
    # sites_file = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/study_22.12_sites.csv"
    # sites_df = pd.read_csv(sites_file)
    # sites_gdf = gpd.GeoDataFrame(
    #     sites_df,
    #     geometry=gpd.points_from_xy(sites_df['site_lon'], sites_df['site_lat']),
    #     crs="EPSG:4326" # As specified in the metadata
    # )
    # sites_gdf = sites_gdf.drop(columns = ['site_name',
    #                                         'model_vs30','class',
    #                                         'Z1.0', 'Z2.5'])
    # sites_gdf = sites_gdf.to_crs(epsg=32611) # Convert to the same CRS as all_data_gdf
    # test_data_gdf = gpd.sjoin_nearest(test_data_gdf, sites_gdf, how="left", distance_col="dist")

    # test_data_df = test_data_gdf.drop(columns=['geometry'])
    # if 'index_right' in test_data_df.columns:
    #     test_data_df.drop(columns=['index_right'], inplace=True)
    pred_file = os.path.join(prediction_dir, 'exact_gp_pred.csv')
    # pred_df = pd.DataFrame({
    #     'eq_var_id': testing_x[:,0].cpu().numpy().astype(int),
    #     'pred_mean': pred_mean + train_y_mean,
    #     'pred_var': pred_var,
    #     'site_id': testing_site_id
    # })
    pred_df.to_csv(pred_file, index=False)

    likelihood_file = os.path.join(prediction_dir, 'likelihood.csv')
    likelihood_df = pd.DataFrame({
        'noise_covar': [likelihood.noise.item()]
    })
    likelihood_df.to_csv(likelihood_file, index=False)

    # pred_mean_file = os.path.join(prediction_dir, 'exact_gp_pred_mean.npy')
    # pred_var_file = os.path.join(prediction_dir, 'exact_gp_pred_var.npy')
    # torch.save(pred_mean+train_y_mean, pred_mean_file)
    # torch.save(pred_var, pred_var_file)





