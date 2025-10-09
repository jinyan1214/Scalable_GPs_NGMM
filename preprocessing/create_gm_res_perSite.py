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

# from mpi4py import MPI

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
dir_out = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/'
dir_fig = dir_out + 'figures/'

#ground motion database
fn_gm_db = 'gm_db.sqlite'
# fn_gm_db = 'gm_db_ASK14.sqlite'

#create output directory
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)


if os.path.exists(dir_out+fn_gm_db):
    #create database connection for ground motion data
    db_gm_cnx = connect(dir_out+fn_gm_db)
    db_gm_cur = db_gm_cnx.cursor()
    # load station metadata
    query = "SELECT * FROM metadata_sta;"
    df_sta_metadata = pd.read_sql_query(query, db_gm_cnx)
    if flag_reset:
        # Remove the old data_sa_scec and data_sa_res tables if they exist
        db_gm_cur.execute("DROP TABLE IF EXISTS data_sa_scec;")
        db_gm_cur.execute("DROP TABLE IF EXISTS data_sa_res;")
        for s_id in df_sta_metadata.site_id.values:
            db_gm_cur.execute(f"DROP TABLE IF EXISTS data_sa_scec_{s_id};")
            db_gm_cur.execute(f"DROP TABLE IF EXISTS data_sa_res_{s_id};")
        db_gm_cnx.commit()
        flag_metadata = True        #store metadata
    else:
        flag_metadata = False
else:
    print(f'ERROR: {dir_out+fn_gm_db} does not exist. Please create the database first.')

# engine = create_engine(f'sqlite:///{dir_out+fn_gm_db}')



df_sta_metadata = df_sta_metadata.head(10) # For testing, limit to first 2 stations


## Check if any of the sqlite tables has period different from the periods that the filename suggests
# for j, per in tqdm(enumerate(per2process)):
#     #set up scec database connection
#     db_scec_cnx = connect(dir_db+fn_scec_dbs['T%.2fs'%per])
#     query = "SELECT DISTINCT IM_Period FROM IM_Data;"
#     df_scec_periods = pd.read_sql_query(query, db_scec_cnx)
#     if (df_scec_periods.shape[0] != 1) or (df_scec_periods.IM_Period.values[0] != per):
#         print('ERROR: SCEC database has more than one period or period does not match the one in the filename.')
#         sys.exit(1)

df_gmm_data = {}
for gmm_key in tqdm(gmm_dict, total = len(gmm_dict), desc='Loading GMM data'):
    df_gmm_data_i = pd.read_pickle(dir_out + f'df_gmm_{gmm_key}.pkl')
    df_gmm_data[gmm_key] = df_gmm_data_i.reset_index()

# Calculate residuals for each period
for j ,per in enumerate(per2process):
    gmm_med_col = ['gmm_T%.2fs'%per for per in per2process]
    print('Processing residuals for period T=%.2fsec (%i of %i)'%(per,j+1,len(per2process)))
    #set up scec database connection
    db_scec_cnx = connect(dir_db+fn_scec_dbs['T%.2fs'%per])
    #iterate over sites
    for l, s_id in tqdm(enumerate(df_sta_metadata.site_id.values), 
                        total = len(df_sta_metadata), 
                        desc=f'Processing sites for period T={per:.2f}s ({j+1} of {len(per2process)})'):
        #create scec sim table
        create_table_scec_data = f"""CREATE TABLE IF NOT EXISTS data_sa_scec_{s_id} (
                                            'scen_id'          integer,
                                            'rup_var_id'       integer,
                                            'period'           real,
                                            'psa'              real,
                                            PRIMARY KEY (scen_id, rup_var_id, period),
                                            FOREIGN KEY (scen_id) REFERENCES metadata_scen (scen_id)
                                        );"""
        if flag_metadata : db_gm_cur.execute(create_table_scec_data);

        #create residuals table
        create_table_res_data = f"""CREATE TABLE IF NOT EXISTS data_sa_res_{s_id} (
                                            'scen_id'          integer,
                                            'rup_var_id'       integer,
                                            'period'           real,
                                            'gmm'              text,
                                            'res'              real,
                                            PRIMARY KEY (scen_id, rup_var_id, period, gmm),
                                            FOREIGN KEY (scen_id)    REFERENCES metadata_scen (scen_id)
                                            FOREIGN KEY (rup_var_id) REFERENCES data_sa_scec  (rup_var_id)
                                            FOREIGN KEY (period)     REFERENCES data_sa_scec  (period)
                                        );"""
        if flag_metadata : db_gm_cur.execute(create_table_res_data);

        db_gm_cnx.commit()  # Commit the changes to the database
        # Load scec data
        time_s = timeit.default_timer()
        query = "SELECT Source_ID, Rupture_ID, Rup_Var_ID, Site_Name, IM_Value FROM IM_Data WHERE Site_Name = '%s'"%(s_id)
        df_scec_data = pd.read_sql_query(query, db_scec_cnx).rename(columns={'Rup_Var_ID':'rup_var_id', 'IM_Value':'psa'})
        print('Time used for loading SCEC data for site %s: %.1f sec'%(s_id, timeit.default_timer() - time_s))
        time_s = timeit.default_timer()
        for k , gmm_key in enumerate(gmm_dict):
            #add scenario id and psa gmm information
            df_scec_data_res = df_scec_data.merge(df_gmm_data[gmm_key][['scen_id','source_id','rupture_id','site_id']+[gmm_med_col[j]]], 
                                            left_on=['Source_ID','Rupture_ID','Site_Name'],
                                            right_on=['source_id','rupture_id','site_id'])
            print('Time used for merging SCEC and metadata for site %s gmm %s: %.1f sec'%(s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()
            #add period information
            df_scec_data_res.loc[:,'period'] = per
            #scale scec psa from cm/sec^2 to g
            df_scec_data_res.loc[:,'psa'] *= 1/grav * 0.01

            #compute residuals
            df_scec_data_res.loc[:,'res'] = np.log(df_scec_data_res.psa.values) - np.log(df_scec_data_res.loc[:,gmm_med_col[j]].values)
            #gmm name
            df_scec_data_res.loc[:,'gmm'] = gmm_key
            print('Time used for calculating residual for site %s gmm %s: %.1f sec'%(s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()
            #append to sql database
            if k == 0:
                df_scec_data_res[['scen_id','rup_var_id','period','psa']].to_sql(f'data_sa_scec_{s_id}',      con=db_gm_cnx, index=False, if_exists='append', method='multi', chunksize=5000)
                # df_scec_data_res[['scen_id','rup_var_id','period','psa']].to_sql('data_sa_scec', con=engine, index=False, if_exists='append', method='multi', chunksize=1000)
                df_scec_data = df_scec_data_res[['Source_ID', 'Rupture_ID', 'rup_var_id', 'Site_Name', 'psa']]
                print('Time used for scec to_sql (%.0f) for site %s gmm %s: %.1f sec'%(len(df_scec_data_res), s_id, gmm_key, timeit.default_timer() - time_s))
                time_s = timeit.default_timer()
            df_scec_data_res[['scen_id','rup_var_id','gmm','period','res']].to_sql(f'data_sa_res_{s_id}', con=db_gm_cnx, index=False, if_exists='append', method='multi', chunksize=5000)
            # df_scec_data_res[['scen_id','rup_var_id','gmm','period','res']].to_sql('data_sa_res', con=engine, index=False, if_exists='append', method='multi', chunksize=1000)
            print('Time used for res to_sql (%.0f) for site %s gmm %s: %.1f sec'%(len(df_scec_data_res), s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()
