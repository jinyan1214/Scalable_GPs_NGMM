import torch
import gpytorch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from GPModel import ExactGPModel
import numpy as np

def load_ExactGP_model(model_path, likelihood_path, train_x, train_y, device):
    # load likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(torch.load(likelihood_path, map_location=device))
    # load model
    model_state = torch.load(model_path, map_location=device)
    if 'source_kernel' in model_state:
        source_effect = True
    else:
        source_effect = False
    if 'eta_kernel' in model_state:
        between_event = True
    else:
        between_event = False
    model = ExactGPModel(train_x, train_y, likelihood,
                         between_kernel=between_event,
                         source_effect=source_effect).to(device)
    model.load_state_dict(model_state)
    # Set into eval mode
    model.eval()
    likelihood.eval()
    return model, likelihood

def select_conditional_data_covar(
        train_x, train_y, test_row, model, device, max_points=1000):
    test_x = torch.from_numpy(test_row[['eq_id', 'source_cst_x', 'source_cst_y', 'source_cst_z',
                                            'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
                                            'site_x', 'site_y', 'site_z']].values.astype(np.float32)).unsqueeze(0).to(device)
    train_x = torch.from_numpy(train_x.astype(np.float32)).to(device)
    covar_mat = model.covar_module(train_x, test_x).evaluate().cpu().numpy().flatten()
    max_points = min(max_points, train_x.shape[0])
    selected_idx = np.argpartition(covar_mat, -max_points)[-max_points:]      # indices of top k elements (unsorted)
    conditional_x = train_x[selected_idx, :].to(device)
    conditional_y = torch.from_numpy(train_y[selected_idx].astype(np.float32)).to(device)
    return conditional_x, conditional_y

def select_conditional_data_dist(train_x, train_y, train_site_ids, test_row, site_neighbor_dict,
                                  source_neighbor_dict, device, max_points=1000):
    # print(test_row)
    test_eq_id = str(test_row['eq_id'])
    test_site_id = str(test_row['site_id'])
    train_eq_ids = (train_x[:, 0]//1000).astype(int)
    
    mask_all = np.zeros(train_x.shape[0], dtype=bool)
    for neighbor_eq_id in source_neighbor_dict[test_eq_id]['neighbor_ids']:
        mask_eq = np.logical_and(
            train_eq_ids == neighbor_eq_id,
            np.isin(train_site_ids, site_neighbor_dict[test_site_id]['neighbor_ids'])
        )
        if np.sum(mask_all) < max_points:
            mask_all = np.logical_or(mask_all, mask_eq)
        else:
            break   
    if np.sum(mask_all) < max_points:
        for neighbor_site_id in site_neighbor_dict[test_site_id]['neighbor_ids']:
            mask_site = (train_site_ids == neighbor_site_id)
            if np.sum(mask_all) < max_points:
                mask_all = np.logical_or(mask_all, mask_site)
            else:
                break
    # print(np.sum(mask_all), "conditional training points selected")
    if np.sum(mask_all) == 0:
        print(f"No conditional training data found for test eq_id {test_eq_id} and site_id {test_site_id}, using the first {max_points} training data")
        conditional_x = torch.from_numpy(train_x[:max_points, :].astype(np.float32)).to(device)
        conditional_y = torch.from_numpy(train_y[:max_points].astype(np.float32)).to(device)
        return conditional_x, conditional_y
    elif len(source_neighbor_dict[test_eq_id]['neighbor_ids']) == 0:
        print(f"No conditional training source data found for test eq_id {test_eq_id}, using all training data with conditional sites")
    elif len(site_neighbor_dict[test_site_id]['neighbor_ids']) == 0:
        print(f"No conditional training site data found for test site_id {test_site_id}, using all training data with conditional sources")
    else:
        if np.sum(mask_all) < max_points:
            print(f"Warning: site: {test_site_id} eq: {test_eq_id} number of conditional training points {np.sum(mask_all)} less than {max_points}")
    conditional_x = torch.from_numpy(train_x[mask_all, :].astype(np.float32)).to(device)
    conditional_y = torch.from_numpy(train_y[mask_all].astype(np.float32)).to(device)
    return conditional_x, conditional_y
