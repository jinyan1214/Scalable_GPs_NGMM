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
from training_utils.GPModel import GPModel
import time
from torch.utils.data import TensorDataset, DataLoader
import argparse
from training_utils.utils import set_seed, read_hdf5_with_site_id, predict_dataset
import geopandas as gpd


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
    parser.add_argument('--num_training_eqs', type=int, default=100,
                        help='Number of earthquakes to use for training')
    parser.add_argument('--num_testing_eqs', type=int, default=1000,
                        help='Number of earthquakes to use for testing')
    parser.add_argument('--random_seed', type=int, default=62,
                        help='Seed for random permutation of earthquake IDs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs to train the model')
    
    args = parser.parse_args()
    num_training_eqs = args.num_training_eqs
    num_epochs = args.num_epochs

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
    rup_var_df['eq_var_id'] = rup_var_df.apply(
        lambda row: int(row['eq_id']*1000 + row['rup_var_id']), axis=1)
    testing_eq_var_ids = rup_var_df['eq_var_id'][total_num_eqs//2:].values
    testing_mask = np.isin(all_data_x[:,0], testing_eq_var_ids)
    print(f"Number of testing points: {np.sum(testing_mask)}")

    testing_x = all_data_x[testing_mask, :]
    testing_y = all_data_y[testing_mask]
    testing_site_id = all_data_site_id[testing_mask]
    # Move to device
    testing_x = torch.from_numpy(testing_x).to(device)
    testing_y = torch.from_numpy(testing_y).to(device)
    # Create dataset and dataloader
    test_dataset = DatasetWithIndex(TensorDataset(testing_x, testing_y))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # create the model
    source_normalizer = 92.8112
    site_normalizer = 40.3403
    path_normalizer = 76.3133
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    inducing_points = torch.zeros((2000,7)) # Placeholder for initializing the model
    model = GPModel(inducing_points=inducing_points, 
                    source_normalizer=source_normalizer, 
                    site_normalizer=site_normalizer, 
                    path_normalizer=path_normalizer)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Load the trained model state
    trained_dir = os.path.join(output_dir, 'trained_models', f'{num_training_eqs}_training_eqs_{num_epochs}_epochs')
    if device == torch.device("cuda"):
        model_state_file = os.path.join(trained_dir, 'gpu_model_trained.pth')
        likelihood_state_file = os.path.join(trained_dir, 'gpu_likelihood_trained.pth')
    else:
        model_state_file = os.path.join(trained_dir, 'cpu_model_trained.pth')
        likelihood_state_file = os.path.join(trained_dir, 'cpu_likelihood_trained.pth')
    
    model.load_state_dict(torch.load(model_state_file))
    likelihood.load_state_dict(torch.load(likelihood_state_file))

    # Evaluate the model on the testing data
    model.eval()
    likelihood.eval()

    pred_mean = torch.zeros((len(test_dataset),))
    pred_var = torch.zeros((len(test_dataset),))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, _, idx in tqdm(test_loader, total=len(test_loader), 
                               desc=f"Predicting {len(test_loader.dataset)} samples in batches:"):
            observed_pred = model(x_batch)
            # pred_mean = torch.cat([pred_mean, observed_pred.mean.cpu()])
            # pred_var = torch.cat([pred_var, observed_pred.lazy_covariance_matrix.diagonal().cpu()])
            pred_mean[idx] = observed_pred.mean.cpu()
            pred_var[idx] = observed_pred.lazy_covariance_matrix.diagonal().cpu()
            torch.cuda.empty_cache()
    # pred_mean = pred_mean[1:].numpy()
    # pred_var = pred_var[1:].numpy()
    pred_mean = pred_mean.numpy()
    pred_var = pred_var.numpy()

    # Compute the RMSE on the testing data
    rmse = np.sqrt(np.mean((pred_mean - testing_y.cpu().numpy())**2))
    print(f"RMSE on the testing data: {rmse}")
    # Save the predictions
    prediction_dir = os.path.join(output_dir, f'prediction_outputs_svgp_{device}', 
            f'{num_training_eqs}_training_eqs_{num_epochs}_epochs')
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
    pred_file = os.path.join(prediction_dir, 'svgp_pred.csv')
    pred_df = pd.DataFrame({
        'eq_var_id': testing_x[:,0].cpu().numpy().astype(int),
        'pred_mean': pred_mean + train_y_mean,
        'pred_var': pred_var,
        'site_id': testing_site_id
    })
    pred_df.to_csv(pred_file, index=False)

    likelihood_file = os.path.join(prediction_dir, 'likelihood.csv')
    likelihood_df = pd.DataFrame({
        'noise_covar': [likelihood.noise.item()]
    })
    likelihood_df.to_csv(likelihood_file, index=False)

    pred_mean_file = os.path.join(prediction_dir, 'exact_gp_pred_mean.npy')
    pred_var_file = os.path.join(prediction_dir, 'exact_gp_pred_var.npy')
    torch.save(pred_mean+train_y_mean, pred_mean_file)
    torch.save(pred_var, pred_var_file)

    
