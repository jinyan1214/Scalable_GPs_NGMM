# This script first randomly permutes the eq_id and selects the last --num_testing_eqs as 
# testing earthquakes. It then selects the first --num_training_eqs as the training earthquakes.
# There are 8358 eq_id in total and the first rup_var is taken for each eq_id. 
# (following the fact that no variation will be observed for a given eq_id).

### parallel utilities
from joblib import Parallel, delayed
import contextlib
import joblib
import h5py
import ujson as json
### Import necessary libraries
import pandas as pd
import numpy as np
from sqlite3 import connect
import json
import os
import argparse
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import shapely
import shutil
import multiprocessing

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

def create_geojson(src_dir, eq_samples, out_file, make_plot=False):
    """Create a GeoJSON file with the locations of the earthquakes."""
    src_dir = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/ruptures_erf36"
    eq_samples_save = eq_samples[['source_id', 'rupture_id']].copy()
    eq_samples_save = eq_samples_save.drop_duplicates(keep='first').reset_index(drop=True)
    for index, row in tqdm(eq_samples_save.iterrows(), total=len(eq_samples_save), 
                           desc=f"Creating GeoJSON"):
        source_id = int(row['source_id'])
        rupture_id = int(row['rupture_id'])
        source_file = os.path.join(src_dir, f"{source_id}_{rupture_id}.txt")
        with open(source_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]

            # Extract the first line which is the probability
            prob = float(lines[0].split('=')[1].strip())
            # Extract the second line which is the magnitude
            magnitude = float(lines[1].split('=')[1].strip())
            # Extract the fourth line which is the number of rows
            num_rows = int(lines[3].split('=')[1].strip())
            # Extract the fifth line which is the number of columns
            num_cols = int(lines[4].split('=')[1].strip())
            # Approximate the rupture geometry of with the first row
            data = []
            for line in lines[6:6 + num_cols]:
                if line.strip():
                    data.append([float(x) for x in line.split()])
            data = np.array(data)
            lon_lat = data[:, [1, 0]]
            geometry = shapely.geometry.LineString(lon_lat.tolist())
        eq_samples_save.loc[index, 'geometry'] = geometry
        eq_samples_save.loc[index, 'probability'] = prob
        eq_samples_save.loc[index, 'magnitude'] = magnitude

    gdf = gpd.GeoDataFrame(eq_samples_save, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)  # Set the coordinate reference system
    gdf.to_file(out_file, driver='GeoJSON')
    if make_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, column='magnitude', legend=True, cmap='viridis', markersize=5)
        ax.set_ylim([31.5, 38.0])
        ax.set_xlim([-122.5, -114.5])
        ctx.add_basemap(ax, crs=gdf.crs.to_string(),  source=ctx.providers.CartoDB.PositronNoLabels)
        fig.savefig(out_file.replace('.geojson', '.png'))
        plt.close(fig)
    print(f"GeoJSON file created: {out_file}")

def save_to_hdf5(df, out_file): 
    ### Order the columns and convert to array
    X = df[['eq_var_id','source_cst_x', 'source_cst_y', 
        'source_cst_z', 'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
        'site_x', 'site_y',]].to_numpy()
    rrup = df['Rrup'].to_numpy()
    Y = df['res'].to_numpy()
    site_id = df['site_id'].to_numpy()
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('rrup', data=rrup)
        f.create_dataset('Y', data=Y)
        f.create_dataset('site_id', data=site_id)
    print(f"Data saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format training data for regression with GPyTorch')
    parser.add_argument('--num_training_eqs', type=int, default=100,
                        help='Number of earthquakes to use for training')
    parser.add_argument('--num_testing_eqs', type=int, default=1000,
                        help='Number of earthquakes to use for testing')
    parser.add_argument('--permutation_seed', type=int, default=42,
                        help='Seed for random permutation of earthquake IDs')
    parser.add_argument('--n_var_per_eq', type=str, default="1",
                        help='Number of variations per earthquake to sample')
    parser.add_argument('--num_cores', type=int, default=20,
                        help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--overwrite_testing', type=bool, default=False,
                        help='Whether to overwrite the testing data if it already exists')
    parser.add_argument('--create_geojson', type=bool, default=True,
                        help='Whether to create GeoJSON files for training and testing data')

    num_training_eqs = parser.parse_args().num_training_eqs
    num_testing_eqs = parser.parse_args().num_testing_eqs
    permutation_seed = parser.parse_args().permutation_seed
    n_var_per_eq = parser.parse_args().n_var_per_eq
    if n_var_per_eq.isdecimal():
        n_var_per_eq = int(n_var_per_eq)
    num_cores = parser.parse_args().num_cores
    overwrite_testing = parser.parse_args().overwrite_testing

    ### Load the database file
    rup_var_per_eq_file = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/rup_var_per_eq.json"
    with open(rup_var_per_eq_file, 'r') as f:
        rup_var_per_eq = json.load(f)

    fn_metadata = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/gm_metadata_expanded.csv"
    df_metadata = pd.read_csv(fn_metadata)

    df_metadata = df_metadata.loc[~np.isin(df_metadata.source_id, [1, 13, 41, 188]),:]
    df_metadata = df_metadata.loc[df_metadata.eq_id.astype(str).isin(list(rup_var_per_eq.keys())), :]
    df_metadata.reset_index(drop=True, inplace=True)

    period = 2.0
    gmm = 'ASK14'
    dir_gm_db = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing/'
    fn_gm_db = dir_gm_db + 'gm_db.sqlite'
    out_dir = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/formatted_data"
    out_dir_testing = os.path.join(out_dir, f'testing_{num_testing_eqs}_eqs')
    out_dir_training = os.path.join(out_dir, f"training_{num_training_eqs}_eqs")
    if os.path.exists(out_dir_testing) and overwrite_testing:
        shutil.rmtree(out_dir_testing)
    os.makedirs(out_dir_testing, exist_ok=True)
    if os.path.exists(out_dir_training) and overwrite_testing:
        shutil.rmtree(out_dir_training)
    os.makedirs(out_dir_training, exist_ok=True)

    ### Read the training and testing sites lists
    sites_split_dir = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/sites_train_test_split"
    site_list_train = pd.read_csv(os.path.join(sites_split_dir, 'site_list_train.csv'))
    site_list_test = pd.read_csv(os.path.join(sites_split_dir, 'site_list_test.csv'))
    site_list_train = site_list_train['site_id'].values
    site_list_test = site_list_test['site_id'].values

    ### Number of earthquakes used for regression
    # Note: there are 8358 earthquakes in the dataset
    # this number will be expanded to bigger number when rup_var is included
    # approximately 70 variations per earthquake
    eq_ids = df_metadata['eq_id'].unique()
    eq_ids = sorted(eq_ids)
    np.random.seed(permutation_seed)  # for reproducibility
    eq_ids_permuted = np.random.permutation(eq_ids)

    testing_eq_ids = eq_ids_permuted[-num_testing_eqs:]
    training_eq_ids = eq_ids_permuted[:num_training_eqs]

    df_metadata_testing = df_metadata[df_metadata['eq_id'].isin(testing_eq_ids)]
    df_metadata_testing = df_metadata_testing[['scen_id',
        'eq_id', 'source_cst_x', 'source_cst_y', 'source_cst_z',
        'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
        'site_x', 'site_y', 'Rrup']]
    df_metadata_training = df_metadata[df_metadata['eq_id'].isin(training_eq_ids)]
    df_metadata_training = df_metadata_training[['scen_id',
        'eq_id', 'source_cst_x', 'source_cst_y', 'source_cst_z',
        'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
        'site_x', 'site_y', 'Rrup']]
    
    df_metadata_selected = pd.concat([df_metadata_training, df_metadata_testing], ignore_index=True)
    # Drop duplicated (this is only useful when num_testing_eqs + num_training_eqs > total number of eqs)
    df_metadata_selected = df_metadata_selected.drop_duplicates(keep='first')
    
    ## Get the first rup_var_id for each eq_id
    # training rup_var_ids
    rup_var_sample = {}
    for eq_id in training_eq_ids:
        eq_id = str(eq_id)
        if n_var_per_eq == 'all':
            rup_var_sample[eq_id] = list(rup_var_per_eq[eq_id])
            continue
        if len(rup_var_per_eq[eq_id]) < n_var_per_eq:
            raise ValueError(f"eq_id {eq_id} has less than {n_var_per_eq} in rup_var_per_eq.json")
        if eq_id in rup_var_per_eq:
            rup_var_sample[eq_id] = list(rup_var_per_eq[eq_id])[:n_var_per_eq]
        else:
            raise ValueError(f"eq_id {eq_id} not found in rup_var_per_eq.json")
    rup_var_sample = pd.DataFrame([(key, value) for key, values in rup_var_sample.items() for value in values], columns=['eq_id', 'rup_var_id'])
    rup_var_sample['eq_id'] = rup_var_sample['eq_id'].astype(int)
    rup_var_sample['rup_var_id'] = rup_var_sample['rup_var_id'].astype(int)

    src_rup_id = df_metadata[['eq_id', 'source_id', 'rupture_id']].drop_duplicates(keep='first')
    rup_var_sample = rup_var_sample.merge(src_rup_id, on='eq_id', how='left')
    rup_var_sample.to_csv(os.path.join(out_dir_training, 'rup_var_sample_training.csv'), index=False)
    training_eq_sample = rup_var_sample.copy()
    # testing rup_var_ids
    rup_var_sample = {}
    for eq_id in testing_eq_ids:
        eq_id = str(eq_id)
        if n_var_per_eq == 'all':
            rup_var_sample[eq_id] = list(rup_var_per_eq[eq_id])
            continue
        if len(rup_var_per_eq[eq_id]) < n_var_per_eq:
            print(f"eq_id {eq_id} has less than {n_var_per_eq} in rup_var_per_eq.json")
            rup_var_sample[eq_id] = list(rup_var_per_eq[eq_id])
        if eq_id in rup_var_per_eq:
            rup_var_sample[eq_id] = list(rup_var_per_eq[eq_id])[:n_var_per_eq]
        else:
            raise ValueError(f"eq_id {eq_id} not found in rup_var_per_eq.json")
    rup_var_sample = pd.DataFrame([(key, value) for key, values in rup_var_sample.items() for value in values], columns=['eq_id', 'rup_var_id'])
    rup_var_sample['eq_id'] = rup_var_sample['eq_id'].astype(int)
    rup_var_sample['rup_var_id'] = rup_var_sample['rup_var_id'].astype(int)

    src_rup_id = df_metadata[['eq_id', 'source_id', 'rupture_id']].drop_duplicates(keep='first')
    rup_var_sample = rup_var_sample.merge(src_rup_id, on='eq_id', how='left')
    rup_var_sample.to_csv(os.path.join(out_dir_testing, 'rup_var_sample_testing.csv'), index=False)
    testing_eq_sample = rup_var_sample.copy()

    # Create a geojson for the training and testing earthquakes
    if parser.parse_args().create_geojson:
        src_dir = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/ruptures_erf36"
        create_geojson(src_dir, training_eq_sample, 
                    os.path.join(out_dir_training, 'training_eqs.geojson'), make_plot=True)
        testing_geojson = os.path.join(out_dir_testing, 'testing_eqs.geojson')
        if overwrite_testing or not os.path.exists(testing_geojson):
            create_geojson(src_dir, testing_eq_sample, testing_geojson, make_plot=True)

    ## create hdf5 files for training and testing data
    training_eq_sample = training_eq_sample.drop(columns=['source_id', 'rupture_id'])
    testing_eq_sample = testing_eq_sample.drop(columns=['source_id', 'rupture_id'])

    rup_var_sample = pd.concat([training_eq_sample, testing_eq_sample], ignore_index=True)
    # Drop duplicated (this is only useful when num_testing_eqs + num_training_eqs > total number of eqs)
    rup_var_sample = rup_var_sample.drop_duplicates(keep='first')
    ### Define the single task function
    def single_task(site_id):
        # Load residual for the site
        global fn_gm_db, period, gmm, training_eq_ids, testing_eq_ids
        global df_metadata_selected, n_var_per_eq
        query = f"""
        SELECT scen_id, rup_var_id, res 
        FROM data_sa_res_{site_id} 
        WHERE period = {period} AND gmm = '{gmm}'
        """
        db_gm_cnx = connect(fn_gm_db)
        res_i = pd.read_sql_query(query, db_gm_cnx)
        # Select scen_id from selected eq and add the location of the earthquake and site
        res_i = res_i.merge(df_metadata_selected, on='scen_id', how='inner')
        # Filter only the sampled rup_var_id
        res_i = res_i.merge(rup_var_sample, on=['eq_id', 'rup_var_id'], how='inner')
        # Combine the eq_id and rup_var_id into eq_var_id
        res_i['eq_var_id'] = res_i['eq_id'] * 1000 + res_i['rup_var_id']
        # Drop the unnecessary columns
        res_i = res_i.drop(columns=['scen_id', 'rup_var_id'])
        # Add site_id to the dataframe for prediction reference
        res_i['site_id'] = site_id
        return res_i


    ### Define the number of threads
    print(f"Using cores: {num_cores}")
    print(f"Number of cores available: {multiprocessing.cpu_count()}")

    ### Process the training sites in parallel
    site_list_train = site_list_train.tolist()  # Convert to list for joblib compatibility
    with tqdm_joblib(tqdm(desc="Loop through training sites", total=len(site_list_train))) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(single_task)(
            site_id
        ) for site_id in site_list_train)
    # print(len(results), "results obtained from training sites")
    ### Concatenate the results
    df_results = pd.concat(results, ignore_index=True)
    # Test if there are duplicated rows
    if df_results.duplicated().any():
        print("Duplicated rows found in trainging eqs results data")
    # print(df_results.head(3))
    df_training = df_results[df_results['eq_id'].isin(training_eq_ids)]
    df_testing = df_results[df_results['eq_id'].isin(testing_eq_ids)]
    # print(df_training.shape)
    # print(df_testing.shape)

    ### Save the results
    file_name = f"training_sites_training_eqs_{period:0.2f}_{gmm}" + \
        f"_{num_training_eqs}_eqs_{n_var_per_eq}_var_per_eq.h5"
    out_file = os.path.join(out_dir_training, file_name)
    if parser.parse_args().overwrite_testing or not os.path.exists(out_file):
        save_to_hdf5(df_training, out_file)
    else:
        print(f"Training earthquake data already exists at {out_file}, skipping save.")

    file_name = f"training_sites_testing_eqs_{period:0.2f}_{gmm}" + \
        f"_{num_testing_eqs}_eqs_{n_var_per_eq}_var_per_eq.h5"
    out_file = os.path.join(out_dir_testing, file_name)
    if parser.parse_args().overwrite_testing or not os.path.exists(out_file):
        save_to_hdf5(df_testing, out_file)
    else:
        print(f"Testing earthquake data already exists at {out_file}, skipping save.")
    
    ### Process the test sites in parallel
    site_list_test = site_list_test.tolist()  # Convert to list for joblib compatibility
    with tqdm_joblib(tqdm(desc="Loop through test sites", total=len(site_list_test))) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(single_task)(
            site_id
        ) for site_id in site_list_test)

    ### Concatenate the results
    df_results = pd.concat(results, ignore_index=True)
    ### Save the results
    df_training = df_results[df_results['eq_id'].isin(training_eq_ids)]
    df_testing = df_results[df_results['eq_id'].isin(testing_eq_ids)]

    # Test if there are duplicated rows
    if df_results.duplicated().any():
        print("Duplicated rows found in results data")

    ### Save the results
    file_name = f"testing_sites_training_eqs_{period:0.2f}_{gmm}" + \
        f"_{num_training_eqs}_eqs_{n_var_per_eq}_var_per_eq.h5"
    out_file = os.path.join(out_dir_training, file_name)
    if parser.parse_args().overwrite_testing or not os.path.exists(out_file):
        save_to_hdf5(df_training, out_file)
    else:
        print(f"Training earthquake data already exists at {out_file}, skipping save.")

    file_name = f"testing_sites_testing_eqs_{period:0.2f}_{gmm}" + \
        f"_{num_testing_eqs}_eqs_{n_var_per_eq}_var_per_eq.h5"
    out_file = os.path.join(out_dir_testing, file_name)
    if parser.parse_args().overwrite_testing or not os.path.exists(out_file):
        save_to_hdf5(df_testing, out_file)
    else:
        print(f"Testing earthquake data already exists at {out_file}, skipping save.")