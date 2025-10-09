### parallel utilities
from joblib import Parallel, delayed
import multiprocessing
import time
import contextlib
import joblib
from tqdm import tqdm
from functools import reduce
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

### Import necessary libraries
import pandas as pd
import numpy as np
from sqlite3 import connect
import ujson as json

### Load the database file
fn_metadata = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing/gm_metadata.csv"
df_metadata = pd.read_csv(fn_metadata)
df_metadata = df_metadata.loc[~np.isin(df_metadata.source_id, [1, 13, 41, 188]),:]
df_metadata.reset_index(drop=True, inplace=True)
period = 2.0
gmm = 'ASK14'
dir_gm_db = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing/'
fn_gm_db = dir_gm_db + 'gm_db.sqlite'

### Define the sites to process
site_list = df_metadata.site_id.unique()
# site_list = site_list[:5] # for testing

### Define the single task function
def single_task(site_id):
    global fn_gm_db, period, gmm
    query = f"SELECT scen_id, rup_var_id, res FROM data_sa_res_{site_id} WHERE period = {period} AND gmm = '{gmm}'"
    db_gm_cnx = connect(fn_gm_db)
    df = pd.read_sql_query(query, db_gm_cnx)
    db_gm_cnx.close()
    df = df.merge(df_metadata[['scen_id','eq_id']], on='scen_id', how='left')
    return df[['eq_id', 'rup_var_id']]

### Count the number of threads
print(f'cpu count: {multiprocessing.cpu_count()}')
num_cores = 20
print(num_cores)

### Run the tasks in parallel
N = len(site_list)
with tqdm_joblib(tqdm(desc="Calculate in parallel", total=N)) as progress_bar:
    results = Parallel(n_jobs=num_cores)(delayed(single_task)(
        site_id
    ) for site_id in site_list)

### Find the unique rup_var_id for each eq_id in the first site
dict_results = results[0].groupby('eq_id')['rup_var_id'].apply(set).to_dict()
for df in tqdm(results[1:], desc="Merging results", total=len(results)-1):
    dict_results_i = df.groupby('eq_id')['rup_var_id'].apply(set).to_dict()
    for eq_id, rup_var_ids in dict_results_i.items():
        if eq_id in dict_results:
            dict_results[eq_id] = dict_results[eq_id].union(rup_var_ids)
        else:
            dict_results[eq_id] = rup_var_ids
for eq_id, rup_var_ids in dict_results.items():
    dict_results[eq_id] = list(rup_var_ids)

with open("/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/rup_var_per_eq.json", "w") as f:
    json.dump(dict_results, f)