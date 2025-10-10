import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pandas as pd
import os
import pygmm
from scipy.interpolate import interp1d
import scipy.stats
import sys
from sqlite3 import connect
sys.path.append("/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src")
from training_utils.utils import read_hdf5, read_hdf5_with_site_id, str2bool
import glob
import argparse
from tqdm import tqdm

def calculate_hazard_for_site(
    site_csv,
    metadata,
    fn_gm_db,
    output_model_dir,
    period,
    ln_sa_values,
    gmm,
    overwrite
):
    site_id = os.path.basename(site_csv).split(".csv")[0].split("_")[1]
    output_site_dir = os.path.join(output_model_dir, site_id)
    os.makedirs(output_site_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_site_dir, "hazard_curve.csv")) and \
       os.path.exists(os.path.join(output_site_dir, "predictions.csv")) and \
       os.path.exists(os.path.join(output_site_dir, "hazard_curve.png")) and \
       os.path.exists(os.path.join(output_site_dir, "residuals_comparison.png")) and \
       (not overwrite):
        print(f"Site {site_id} already processed, skipping...")
        return None
    prediction_i = pd.read_csv(site_csv)
    # legacy naming convention
    if 'eq_var_id' in prediction_i.columns:
        prediction_i = prediction_i.rename(columns={'eq_var_id': 'eq_id'})
    # Make gmm predictions
    metadata_subset = metadata[metadata['site_id'] == site_id].copy()
    # for i, row in prediction_i.iterrows():
    for i, row in tqdm(prediction_i.iterrows(), total=prediction_i.shape[0], leave=False):
        gm_data = metadata_subset[
            (metadata_subset['eq_id'] == row['eq_id']) & 
            (metadata_subset['site_id'] == site_id)
        ].iloc[0]
        scen = pygmm.model.Scenario(
            mag=gm_data['mag'],
            dip=gm_data['dip'],
            mechanism='SS',
            depth_tor=gm_data['z_tor'],
            width=gm_data['width'],
            dist_rup=gm_data['Rrup'],
            dist_jb=gm_data['Rjb'],
            dist_x=gm_data['Rx'],
            dist_y0=gm_data['Ry0'],
            v_s30=gm_data['vs30_scec'],
            depth_1_0=gm_data['z1.0']
        )
        gmm_obj = gmm(scen)
        prediction_i.loc[i, 'gmm_mean'] = gmm_obj.interp_spec_accels(period)
        prediction_i.loc[i, 'gmm_ln_std'] = gmm_obj.interp_ln_stds(period)
        prediction_i.loc[i, 'gmm_ln_tau'] = interp1d(
            np.log(gmm_obj.periods), gmm_obj._tau[gmm_obj.INDICES_PSA], kind='linear',
            copy = False, bounds_error = False, fill_value = np.nan)(np.log(period)   
        )
        prediction_i.loc[i, 'gmm_ln_phi'] = interp1d(
            np.log(gmm_obj.periods), gmm_obj._phi[gmm_obj.INDICES_PSA], kind='linear',
            copy = False, bounds_error = False, fill_value = np.nan)(np.log(period)
        )
        prediction_i.loc[i, 'probability'] = gm_data['probability']
    # Get the cybershake simulated ground motion
    query = f"""
            SELECT scen_id, rup_var_id, res 
            FROM data_sa_res_{site_id} 
            WHERE period = {2.0} AND gmm = 'ASK14'
            """
    db_gm_cnx = connect(fn_gm_db)
    cybershake_res = pd.read_sql_query(query, db_gm_cnx)
    db_gm_cnx.close()
    cybershake_res = cybershake_res.merge(metadata_subset[['scen_id', 'eq_id']], on = 'scen_id')
    cybershake_res['cybershake_sim_var'] = cybershake_res.groupby('eq_id')['res'].apply(np.var)
    cybershake_first_rup_var = cybershake_res[cybershake_res['rup_var_id']==0]
    cybershake_first_rup_var = cybershake_first_rup_var.rename(columns={'res': 'cybershake_res'})
    prediction_i = prediction_i.merge(cybershake_first_rup_var[['eq_id', 'cybershake_res']], on = 'eq_id')

    # Save the prediction with GMM and cybershake results
    prediction_i.to_csv(os.path.join(output_site_dir, "predictions.csv"), index = False)

    # Make hazard curves
    ngmm_poe_values = np.zeros_like(ln_sa_values)
    gmm_poe_values = np.zeros_like(ln_sa_values)
    ngmm_alea_only_poe_values = np.zeros_like(ln_sa_values)
    cybershake_training_rup_var_poe_values = np.zeros_like(ln_sa_values)
    for i, row in prediction_i.iterrows():
        gmm_poe_values += row['probability'] * (1 - scipy.stats.norm.cdf(
            ln_sa_values, loc=np.log(row['gmm_mean']), 
            scale=row['gmm_ln_std'])
        )

        ngmm_poe_values += row['probability'] * (1 - scipy.stats.norm.cdf(
            ln_sa_values, loc=np.log(row['gmm_mean'])+row['mean'],
            scale=np.sqrt(row['variance_with_noise']))     
        )

        ngmm_alea_only_poe_values += row['probability'] * (1 - scipy.stats.norm.cdf(
            ln_sa_values, loc=np.log(row['gmm_mean'])+row['mean'],
            scale=np.sqrt(row['variance_with_noise'] - row['variance']))
        )
        cybershake_training_rup_var_poe_values += row['probability'] * (
            (np.log(row['gmm_mean']) + row['cybershake_res'] > ln_sa_values).astype(float)
        )
    # Save the hazard curves
    hazard_curve_df = pd.DataFrame({
        'ln_sa': ln_sa_values,
        'gmm_poe': gmm_poe_values,
        'ngmm_poe': ngmm_poe_values,
        'ngmm_alea_only_poe': ngmm_alea_only_poe_values,
        'cybershake_training_rup_var_poe': cybershake_training_rup_var_poe_values
    })
    hazard_curve_df.to_csv(os.path.join(output_site_dir, "hazard_curve.csv"), index = False)
    # Plot and save the hazard curves
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(np.exp(ln_sa_values), ngmm_poe_values, label='NGMM')
    ax.plot(np.exp(ln_sa_values), ngmm_alea_only_poe_values, label='NGMM (Aleatory only)')
    ax.plot(np.exp(ln_sa_values), gmm_poe_values, label='GMM')
    ax.plot(np.exp(ln_sa_values), cybershake_training_rup_var_poe_values, label='Cybershake (Training ruptures)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('Spectral Acceleration 2.0 (g)')
    ax.set_ylabel('Annual Probability of Exceedance')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.set_title(f'Site {site_id}')
    ax.set_ylim([1e-6, 1e-1])
    fig.savefig(os.path.join(output_site_dir, "hazard_curve.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    # Plot and save the ngmm vs cybershake residuals
    fig, ax = plt.subplots(1, 2, figsize=(8,3))
    ax[0].plot(prediction_i['mean'], prediction_i['cybershake_res'], '.')
    ax[0].plot([-1, 3], [-1, 3], 'k--')
    ax[0].set_xlabel('NGMM Residual Prediction')
    ax[0].set_ylabel('Cybershake Residual')

    ax[1].plot(prediction_i['gmm_ln_std'], np.sqrt(prediction_i['variance_with_noise']), '.')
    ax[1].plot([0.5, 0.85], [0.5, 0.85], 'k--')
    ax[1].set_xlabel('GMM ln_std')
    ax[1].set_ylabel('NGMM Total ln_std')
    fig.savefig(os.path.join(output_site_dir, "residuals_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return None

def get_site_var(site_id,
                 output_model_dir):
    output_site_dir = os.path.join(output_model_dir, site_id)
    prediction_i = pd.read_csv(os.path.join(output_site_dir, "predictions.csv"))
    # If weighted average using rupture probabilities as weights
    # weights = prediction_i['probability'].values/np.sum(prediction_i['probability'].values)
    # weighted_var_ngmm = np.power(weights, 1).dot(prediction_i['variance_with_noise'])
    # weighted_var_gmm = np.power(weights, 1).dot(np.power(prediction_i['gmm_ln_std'], 2))
    # empirical_var_gmm = weights.dot(np.power(prediction_i['cybershake_res'], 2))
    # If simple average
    var_ngmm = prediction_i['variance_with_noise'].mean()
    var_gmm = np.power(prediction_i['gmm_ln_std'], 2).mean()
    empirical_var_gmm = np.power(prediction_i['cybershake_res'], 2).mean()

    mean_ngmm = prediction_i['mean'].mean()
    res_gmm = prediction_i['cybershake_res'].mean()

    return var_ngmm, var_gmm, empirical_var_gmm, mean_ngmm, res_gmm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format training data for regression with GPyTorch')
    parser.add_argument('--prediction_dir', type=str,
                        help='Directory to save prediction results')
    parser.add_argument('--dir_suffix', type=str, default="_2.00_ASK14_4179_eqs_1_var_per_eq",
                        help='Suffix for the prediction directory')
    parser.add_argument('--use_MPI', type=str2bool, default=False,
                        help='Whether to use MPI for parallel processing')
    parser.add_argument('--overwrite', type=str2bool, default=False,
                        help='Whether to overwrite existing output files')
    parser.add_argument('--prediction_eq_type', type=str, default="testing",
                        help='Type of prediction equations to use')

    args = parser.parse_args()
    prediction_dir = args.prediction_dir
    dir_suffix = args.dir_suffix
    use_MPI = args.use_MPI
    overwrite = args.overwrite
    prediction_eq_type = args.prediction_eq_type

    if use_MPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    
    # Define some data dir
    fn_metadata = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/gm_metadata_expanded.csv"
    sites_file = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/study_22.12_sites.csv"
    dir_gm_db = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing/'
    fn_gm_db = dir_gm_db + 'gm_db.sqlite'

    # Load metadata and site info
    metadata = pd.read_csv(fn_metadata)
    sites_df = pd.read_csv(sites_file)

    # Define some model parameters
    gmm = pygmm.AbrahamsonSilvaKamai2014
    period = 2.0
    output_dir = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/hazard_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_model_dir = os.path.join(output_dir, prediction_dir.split("/")[-1])
    os.makedirs(output_model_dir, exist_ok=True)
    output_model_dir = os.path.join(output_model_dir, prediction_eq_type)
    os.makedirs(output_model_dir, exist_ok=True)
    # Define the sa to plot
    ln_sa_values = np.log(np.linspace(0.01, 3, 1000))

    # Loop over test sites
    test_sites_dirname = f"testing_sites_{prediction_eq_type}_eqs" + dir_suffix
    test_sites_dir = os.path.join(prediction_dir, test_sites_dirname)
    test_sites_csvs = glob.glob(os.path.join(test_sites_dir, "*.csv"))

    if rank == 0:
        print(f"Start to calculate hazard for {len(test_sites_csvs)} testing sites using {size} processes")
    if use_MPI:
        comm.Barrier()
    for i, test_site_csv in tqdm(enumerate(test_sites_csvs),
                                 total=len(test_sites_csvs), disable=(rank!=0)):
        if i % size != rank:
            continue
        calculate_hazard_for_site(
            test_site_csv,
            metadata,
            fn_gm_db,
            output_model_dir,
            period,
            ln_sa_values,
            gmm,
            overwrite
        )
    
    # Loop over training sites
    train_sites_dirname = f"training_sites_{prediction_eq_type}_eqs" + dir_suffix
    train_sites_dir = os.path.join(prediction_dir, train_sites_dirname)
    train_sites_csvs = glob.glob(os.path.join(train_sites_dir, "*.csv"))


    if rank == 0:
        print(f"Start to calculate hazard for {len(train_sites_csvs)} training sites using {size} processes")
    if use_MPI:
        comm.Barrier()
    for i, train_site_csv in tqdm(enumerate(train_sites_csvs),
                                 total=len(train_sites_csvs), disable=(rank!=0)):
        if i % size != rank:
            continue
        calculate_hazard_for_site(
            train_site_csv,
            metadata,
            fn_gm_db,
            output_model_dir,
            period,
            ln_sa_values,
            gmm,
            overwrite
        )

    if use_MPI:
        comm.Barrier()
    
    sites_df = sites_df.set_index('site_id')
    ngmm_var = np.zeros((sites_df.shape[0],))
    gmm_var = np.zeros((sites_df.shape[0],))
    gmm_empirical_var = np.zeros((sites_df.shape[0],))
    mean_ngmm = np.zeros((sites_df.shape[0],))
    res_gmm = np.zeros((sites_df.shape[0],))

    if rank == 0:
        print(f"Start to collect site variances for {sites_df.shape[0]} sites using {size} processes")
    if use_MPI:
        comm.Barrier()
    for i, site_id in tqdm(enumerate(sites_df.index),
                           total=sites_df.shape[0], disable=(rank!=0)):
        if i % size != rank:
            continue
        site_file = os.path.join(output_model_dir, site_id, "predictions.csv")
        if os.path.exists(site_file):
            ngmm_var_i, gmm_var_i, gmm_empirical_var_i, mean_ngmm_i, res_gmm_i = get_site_var(site_id, output_model_dir)
            ngmm_var[i] = ngmm_var_i
            gmm_var[i] = gmm_var_i
            gmm_empirical_var[i] = gmm_empirical_var_i
            mean_ngmm[i] = mean_ngmm_i
            res_gmm[i] = res_gmm_i
        else:
            print(f"Site file {site_file} does not exist, skipping...")
            ngmm_var[i] = np.nan
            gmm_var[i] = np.nan
            gmm_empirical_var[i] = np.nan
            mean_ngmm[i] = np.nan
            res_gmm[i] = np.nan

    if use_MPI:
        comm.Barrier()
        if rank == 0:
            ngmm_var_all = np.zeros((sites_df.shape[0],))
            gmm_var_all = np.zeros((sites_df.shape[0],))
            gmm_empirical_var_all = np.zeros((sites_df.shape[0],))
            mean_ngmm_all = np.zeros((sites_df.shape[0],))
            res_gmm_all = np.zeros((sites_df.shape[0],))
        else:
            ngmm_var_all = None
            gmm_var_all = None
            gmm_empirical_var_all = None
            mean_ngmm_all = None
            res_gmm_all = None
        comm.Reduce(ngmm_var, ngmm_var_all, op=MPI.SUM, root=0)
        comm.Reduce(gmm_var, gmm_var_all, op=MPI.SUM, root=0)
        comm.Reduce(gmm_empirical_var, gmm_empirical_var_all, op=MPI.SUM, root=0)
        comm.Reduce(mean_ngmm, mean_ngmm_all, op=MPI.SUM, root=0)
        comm.Reduce(res_gmm, res_gmm_all, op=MPI.SUM, root=0)
        if rank == 0:
            sites_df['ngmm_var'] = ngmm_var_all
            sites_df['gmm_var'] = gmm_var_all
            sites_df['gmm_empirical_var'] = gmm_empirical_var_all
            sites_df['mean_ngmm'] = mean_ngmm_all
            sites_df['res_gmm'] = res_gmm_all
            sites_df.to_csv(os.path.join(output_model_dir, "site_variances.csv"))
    else:
        sites_df['ngmm_var'] = ngmm_var
        sites_df['gmm_var'] = gmm_var
        sites_df['gmm_empirical_var'] = gmm_empirical_var
        sites_df['mean_ngmm'] = mean_ngmm
        sites_df['res_gmm'] = res_gmm
        sites_df.to_csv(os.path.join(output_model_dir, "site_variances.csv"))
