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

# from sqlalchemy import create_engine

from mpi4py import MPI

start_time = timeit.default_timer()
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

#gravity
grav = 9.81

#reset database
flag_reset = True

#ground motion models to evaluate
gmm_dict = {'ASK14':pygmm.AbrahamsonSilvaKamai2014, 'CY14':pygmm.ChiouYoungs2014}
#gmm_dict = {'ASK14':pygmm.AbrahamsonSilvaKamai2014}

#output directories
dir_out = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/data_perSite'

comm = MPI.COMM_WORLD
numP = comm.Get_size()  # number of processes
proc_id = comm.Get_rank()  # process id

#create output directory
if not os.path.isdir(dir_out):
    print(f'Output dir does not exist: {dir_out}')
    exit(1)

if flag_reset:
    if proc_id == 0:
        #remove all files in the output directory
        for file in glob.glob(os.path.join(dir_out, '*.pkl')):
            os.remove(file)
        print(f'Removed all files in {dir_out}')


# Load sta_metadata
fn_gm_db = 'gm_db.sqlite'
dir_gm_db = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/'
db_gm_cnx = connect(dir_gm_db+fn_gm_db)
query = "SELECT * FROM metadata_sta;"
df_sta_metadata = pd.read_sql_query(query, db_gm_cnx)


# df_sta_metadata = df_sta_metadata.head(30) # For testing, limit to first 2 stations


for i, s_id in tqdm(enumerate(df_sta_metadata.site_id.values),
                    total=len(df_sta_metadata.site_id.values),
                    desc='Processing stations',
                    disable=(proc_id != 0)):
    if i % numP == proc_id:
        for per in per2process:
            #set up scec database connection
            db_scec_cnx = connect(dir_db+fn_scec_dbs['T%.2fs'%per])
            #check if the site exists in the scec database
            query = "SELECT Source_ID, Rupture_ID, Rup_Var_ID, Site_Name, IM_Value FROM IM_Data WHERE Site_Name = '%s'"%(s_id)
            df_scec_data = pd.read_sql_query(query, db_scec_cnx).rename(columns={'Rup_Var_ID':'rup_var_id', 'IM_Value':'psa'})
            pickle_file_name = f'df_scec_{s_id}_T{per:.2f}s.pkl'
            pickle_file_path = os.path.join(dir_out, pickle_file_name)
            df_scec_data.to_pickle(pickle_file_path)