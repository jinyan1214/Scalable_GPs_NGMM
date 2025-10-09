#general libraries
import os
import sys
import glob
import pathlib
import warnings
import timeit
from tqdm import tqdm
#regular expressions
import re
#database libraries
from sqlite3 import connect
#arithmetic libraries
import numpy as np
import scipy
#statistics libraries
import pandas as pd
#ground motion models
import pygmm
#ipython
from IPython.display import display, clear_output

from mpi4py import MPI


#surpress warnings
warnings.filterwarnings('ignore')

# Define Variables
#metadata filenames
fn_metadata = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing/gm_metadata.csv'

#gravity
grav = 9.81;

#sa databases directory and filenaems
dir_db = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/'
fn_scec_dbs = {'T2.00s':  'study_22_12_2.0sec.sqlite',
            'T2.20s':  'study_22_12_2.2sec.sqlite',
            'T2.40s':  'study_22_12_2.4sec.sqlite',
            'T2.60s':  'study_22_12_2.6sec.sqlite',
            'T2.80s':  'study_22_12_2.8sec.sqlite',
            'T3.00s':  'study_22_12_3.0sec.sqlite',
            'T3.50s':  'study_22_12_3.5sec.sqlite',
            'T4.00s':  'study_22_12_4.0sec.sqlite',
            'T4.40s':  'study_22_12_4.4sec.sqlite',
            'T5.00s':  'study_22_12_5.0sec.sqlite',
            'T5.50s':  'study_22_12_5.5sec.sqlite',
            'T6.00s':  'study_22_12_6.0sec.sqlite',
            'T6.50s':  'study_22_12_6.5sec.sqlite',
            'T7.50s':  'study_22_12_7.5sec.sqlite',
            'T8.50s':  'study_22_12_8.5sec.sqlite',
            'T10.00s': 'study_22_12_10.0sec.sqlite'}
# fn_scec_dbs = {'T2.00s':  'study_22_12_lf_6_periods.sqlite',
#                'T3.00s':  'study_22_12_lf_6_periods.sqlite',
#                'T4.00s':  'study_22_12_lf_6_periods.sqlite',
#                'T5.00s':  'study_22_12_lf_6_periods.sqlite',
#                'T7.50s':  'study_22_12_lf_6_periods.sqlite',
#                'T10.00s': 'study_22_12_lf_6_periods.sqlite'}

#periods to process
# per2process = [2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.4,5.0,5.5,6.0,6.5,7.5,8.5,10.0]
# per2process = [2.0,3.0,4.0,5.0,7.5,10.0]
# per2process = [2.0]
#periods to process (in separate batches - testing)
per2process = [2.0,2.2]     #batch 1
# per2process = [2.4,2.6]     #batch 2
#periods to process (in separate batches)
# per2process = [2.0,2.2,2.4,2.6,2.8] #batch 1
# per2process = [3.0,3.5,4.0,4.4]     #batch 2
# per2process = [5.0,5.5,6.0,6.5]     #batch 3
# per2process = [7.5,8.5,10.0]        #batch 4

#reset database
flag_reset = True

#ground motion models to evaluate
gmm_dict = {'ASK14':pygmm.AbrahamsonSilvaKamai2014, 'CY14':pygmm.ChiouYoungs2014}
#gmm_dict = {'ASK14':pygmm.AbrahamsonSilvaKamai2014}

# Output info
#output directories
dir_out = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/'
dir_fig = dir_out + 'figures/'

#ground motion database
fn_gm_db = 'gm_db.sqlite'
# fn_gm_db = 'gm_db_ASK14.sqlite'

## Read Input Data
# metadata dataframe
df_metadata = pd.read_csv(fn_metadata)
# df_metadata = df_metadata.iloc[0:10,:] # for testing purposes.
#filter bad sources
df_metadata = df_metadata.loc[~np.isin(df_metadata.source_id, [1, 13, 41, 188]),:]
df_metadata.reset_index(drop=True, inplace=True)

comm = MPI.COMM_WORLD
numP = comm.Get_size()  # number of processes
proc_id = comm.Get_rank()  # process id

if proc_id == 0:
    ### Procesing
    ## Establish database
    # create output directory
    if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    if os.path.exists(dir_out+fn_gm_db):
        if flag_reset:
            os.remove(dir_out+fn_gm_db) #remove old database if exists
            flag_metadata = True        #store metadata
        else:
            flag_metadata = False
    else:
        flag_metadata = False
comm.Barrier()  # Ensure all processes reach this point before proceeding
# create database connection for ground motion data
db_gm_cnx = connect(dir_out+fn_gm_db)
db_gm_cur = db_gm_cnx.cursor()

## Parse event and station metadata
# earthquake metadata
_, i_eq = np.unique(df_metadata.eq_id, return_index=True)
col_eq  = ['eq_id', 'source_id', 'rupture_id', 'mag', 'dip', 'rake', 'z_tor', 'width', 'utm_zone',
        'source_mpt_lat', 'source_mpt_lon', 'source_mpt_x', 'source_mpt_y', 'source_mpt_z'] 
df_eq_metadata = df_metadata.loc[i_eq,col_eq].reset_index(drop=True)

# site metadata
_, i_sta = np.unique(df_metadata.site_id, return_index=True)
col_sta  = ['site_id', 'site_name','vs30_thompson', 'vs30_scec', 'z1.0', 'z2.5', 'utm_zone','site_lat', 'site_lon', 'site_x', 'site_y']
df_sta_metadata = df_metadata.loc[i_sta,col_sta].reset_index(drop=True)

if proc_id == 0:
    ## Create event and station tables
    #create earthquake data
    create_table_eq_data = """CREATE TABLE IF NOT EXISTS metadata_eq (
                                        'eq_id'          integer PRIMARY KEY,
                                        'source_id'      integer,
                                        'rupture_id'     integer,
                                        'mag'            real,
                                        'dip'            real,
                                        'rake'           real,
                                        'z_tor'          real,
                                        'width'          real,
                                        'utm_zone'       text,
                                        'source_mpt_lat' real,
                                        'source_mpt_lon' real,
                                        'source_mpt_x'   real,
                                        'source_mpt_y'   real,
                                        'source_mpt_z'   real
                                    );"""
    db_gm_cur.execute(create_table_eq_data)
    #add earthquake metadata
    if flag_metadata: df_eq_metadata.to_sql('metadata_eq', con=db_gm_cnx, index=False, if_exists='append')
    #create station data
    create_table_sta_data = """CREATE TABLE IF NOT EXISTS metadata_sta (
                                        'site_id'        text PRIMARY KEY,
                                        'site_name'      text,
                                        'vs30_thompson'  real,
                                        'vs30_scec'      real,
                                        'z1.0'           real,
                                        'z2.5'           real,
                                        'utm_zone'       text,
                                        'site_lat'       real,
                                        'site_lon'       real,
                                        'site_x'         real,
                                        'site_y'         real
                                    );"""
    db_gm_cur.execute(create_table_sta_data)
    #add station metadata
    if flag_metadata: df_sta_metadata.to_sql('metadata_sta', con=db_gm_cnx, index=False, if_exists='append');

    ## Parse scenarios metadata
    col_scen = ['scen_id', 'eq_id', 'site_id',
                'Rrup', 'Rfsrc', 'Rjb', 'Rx', 'Ry0', 'utm_zone',
                'source_cst_lat','source_cst_lon', 'source_cst_x', 'source_cst_y', 'source_cst_z']
    df_gm_metadata = df_metadata[col_scen].copy()

    ## Create Scenario table
    #create scenario data
    create_table_scen_data = """CREATE TABLE IF NOT EXISTS metadata_scen (
                                        'scen_id'        integer PRIMARY KEY,
                                        'eq_id'          integer,
                                        'site_id'        text,
                                        'Rrup'           real,
                                        'Rfsrc'          real,
                                        'Rjb'            real,
                                        'Rx'             real,
                                        'Ry0'            real,
                                        'utm_zone'       text,
                                        'source_cst_lat' real,
                                        'source_cst_lon' real,
                                        'source_cst_x'   real,
                                        'source_cst_y'   real,
                                        'source_cst_z'   real,
                                        FOREIGN KEY (eq_id)   REFERENCES metadata_eq  (eq_id),
                                        FOREIGN KEY (site_id) REFERENCES metadata_sta (site_id)
                                    );"""
    db_gm_cur.execute(create_table_scen_data)

    #add station metadata
    if flag_metadata: df_gm_metadata.to_sql('metadata_scen', con=db_gm_cnx, index=False, if_exists='append')

    del df_gm_metadata

    #create earthquake data
    create_table_gmm_data = """CREATE TABLE IF NOT EXISTS data_sa_gmm (
                                        'scen_id'        integer,
                                        'period'         real,
                                        'gmm'            text,
                                        'psa'            real,
                                        PRIMARY KEY (scen_id, period, gmm),
                                        FOREIGN KEY (scen_id) REFERENCES metadata_scen (scen_id)
                                    );"""
    db_gm_cur.execute(create_table_gmm_data)
    ## GMM Ground Motions
    # Compute median ground motions for each scenario
    # create dataframe for ground motion predictions
    df_gmm_data = {key:df_metadata.copy() for key in gmm_dict}
else:
    df_gmm_data = None

scenarios_to_run_global = np.arange(df_metadata.shape[0])
scenarios_per_proc = df_metadata.shape[0] // numP
remainder_scenarios = df_metadata.shape[0] % numP
if proc_id < remainder_scenarios:
    # If the process id is less than the remainder, it gets one extra scenario
    scenarios_per_proc += 1
scenarios_to_run_local = np.zeros(scenarios_per_proc, dtype=int)

# Scatter the scenarios to all processes, each process runs a chunk of the scenarios
if proc_id == 0:
    scenarios_to_run_local = scenarios_to_run_global[:scenarios_per_proc]
    proc_scenario_l = np.zeros(numP, dtype=int)
    proc_scenario_u = np.zeros(numP, dtype=int)
    start = scenarios_per_proc
    proc_scenario_u[0] = scenarios_per_proc

for i in range(1, numP):
    if proc_id == 0:
        num_scen_target = df_metadata.shape[0] // numP
        if i < remainder_scenarios:
            num_scen_target += 1
        end = start + num_scen_target
        # print(f"Rank 0 sending data: {scenarios_to_run_global[start:end]} to rank {i}")
        comm.send(scenarios_to_run_global[start:end], dest=i, tag=i)
        proc_scenario_l[i] = start
        proc_scenario_u[i] = end
        start = end
    elif proc_id == i:
        scenarios_to_run_local = comm.recv(source=0, tag=i)
print(f"Rank {proc_id} analyze data: {np.min(scenarios_to_run_local)} - {np.max(scenarios_to_run_local)}, scenarios_per_proc: {scenarios_per_proc}, total scenarios: {df_metadata.shape[0]}")
# Initialize a dictionary to hold GMM data for each process
gmm_data_local = {key:np.zeros((scenarios_per_proc, len(per2process))) for key in gmm_dict}  
comm.Barrier()  # Ensure all processes reach this point before proceeding
# Evaluate ground motion models for each scenario
for k, gmm_key in enumerate(gmm_dict):
    # print('Evaluating GMM: %s (%i of %i) ...'%(gmm_key,k+1,len(gmm_dict))+60*' ')
    for ind, j in tqdm(enumerate(scenarios_to_run_local), total=len(scenarios_to_run_local), desc=f'Evaluating scenarios proc: {proc_id}, GMM: {gmm_key}'):
        gm = df_metadata.iloc[j,:].to_dict()  #get ground motion metadata for scenario j
        #define ground motion scenario
        s = pygmm.model.Scenario(mag=gm['mag'], dip=gm['dip'], mechanism='SS',
                                depth_tor=gm['z_tor'],  width=gm['width'],
                                dist_rup=gm['Rrup'], dist_jb=gm['Rjb'],
                                dist_x=gm['Rx'], dist_y0=gm['Ry0'],
                                v_s30=gm['vs30_scec'], depth_1_0=gm['z1.0'])
        #evaluate scenario for periods of interest
        gmm_data_local[gmm_key][ind,:] = gmm_dict[gmm_key](s).interp_spec_accels(per2process)
    
    # Gather the gmm_data for each GMM from all processes
    gmm_data_per_gmm = None
    if proc_id == 0:
        gmm_data_per_gmm = np.zeros((df_metadata.shape[0], len(per2process)))
        gmm_data_per_gmm[0:scenarios_per_proc, :] = gmm_data_local[gmm_key]
    for i in range(1, numP):
        if proc_id == 0:
            # print(f"Rank 0 gathering data from rank {i}")
            gmm_data_per_gmm[proc_scenario_l[i]:proc_scenario_u[i], :] = comm.recv(source=i, tag=i)
        elif proc_id == i:
            # print(f"Rank {proc_id} sending data to rank 0")
            comm.send(gmm_data_local[gmm_key], dest=0, tag=proc_id)
    # comm.Gather(gmm_data_local[gmm_key], gmm_data_per_gmm, root=0)
    comm.Barrier()  # Ensure all processes reach this point before proceeding
    # Save as a dataframe for each GMM
    if proc_id == 0:
        print(f"Gathered GMM data for {gmm_key} from all processes."
              f" Shape of gathered data: {gmm_data_per_gmm.shape}")
        # Store the gmm_data to df_gmm_data
        gmm_med_col = ['gmm_T%.2fs'%per for per in per2process]
        df_gmm_data[gmm_key].loc[:,gmm_med_col] = gmm_data_per_gmm

        # Set the indices for easiy access later
        df_gmm_data[gmm_key] = df_gmm_data[gmm_key].set_index(['source_id','rupture_id','site_id'])
        # Save the GMM dataframe as a pickle for later use in calculating residuals
        df_gmm_data[gmm_key].to_pickle(dir_out + f'df_gmm_{gmm_key}.pkl')
        for j, per in enumerate(per2process):
            #add ground motion data to database
            df_gmm_per_data = df_gmm_data[gmm_key][['scen_id']].copy()
            df_gmm_per_data.loc[:,'gmm']    = gmm_key
            df_gmm_per_data.loc[:,'period'] = per
            df_gmm_per_data.loc[:,'psa']    = df_gmm_data[gmm_key].loc[:,gmm_med_col[j]].values
            # print(df_gmm_per_data)
            df_gmm_per_data.to_sql('data_sa_gmm', con=db_gm_cnx, index=False, if_exists='append')

# if proc_id == 0:
#     query = f"SELECT scen_id, gmm, period, psa FROM data_sa_gmm WHERE scen_id = {16084}"
#     df_gm_data = pd.read_sql_query(query, con=db_gm_cnx)
#     print(df_gm_data)
            