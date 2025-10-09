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
from sqlite3 import connect
import scipy
#statistics libraries
import pandas as pd
#ground motion models
import pygmm
#ipython
from IPython.display import display, clear_output

#gravity
grav = 9.81

#reset database
flag_reset = True

#output directories
dir_out = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/'
dir_scec_extracted = '/resnick/groups/enceladus/glavrent/Scalable_GPs/Data/preprocessing_jz/data_perSite/'
dir_fig = dir_out + 'figures/'
#ground motion database
fn_gm_db = 'gm_db.sqlite'

gmm_dict = {'ASK14':pygmm.AbrahamsonSilvaKamai2014, 'CY14':pygmm.ChiouYoungs2014}

if os.path.exists(dir_out+fn_gm_db):
    #create database connection for ground motion data
    db_gm_cnx = connect(dir_out+fn_gm_db)
    db_gm_cur = db_gm_cnx.cursor()
    # load station metadata
    query = "SELECT * FROM metadata_sta;"
    df_sta_metadata = pd.read_sql_query(query, db_gm_cnx)
    print('Loaded station metadata from database.')
    if flag_reset:
        # Remove the old data_sa_scec and data_sa_res tables if they exist
        db_gm_cur.execute("DROP TABLE IF EXISTS data_sa_scec;")
        db_gm_cur.execute("DROP TABLE IF EXISTS data_sa_res;")
        for s_id in tqdm(df_sta_metadata.site_id.values, 
                         total=len(df_sta_metadata.site_id.values), 
                         desc='Dropping old tables'):
            db_gm_cur.execute(f"DROP TABLE IF EXISTS data_sa_scec_{s_id};")
            db_gm_cur.execute(f"DROP TABLE IF EXISTS data_sa_res_{s_id};")
        db_gm_cnx.commit()
        flag_metadata = True        #store metadata
    else:
        flag_metadata = False
else:
    print(f'ERROR: {dir_out+fn_gm_db} does not exist. Please create the database first.')

# Periods to process
per2process = [2.0,2.2]     #batch 1

# df_sta_metadata = df_sta_metadata.head(10) # For testing, limit to first 2 stations

# Load GMM median data
df_gmm_data = {}
for gmm_key in tqdm(gmm_dict, total = len(gmm_dict), desc='Loading GMM data'):
    df_gmm_data_i = pd.read_pickle(dir_out + f'df_gmm_{gmm_key}.pkl')
    df_gmm_data[gmm_key] = df_gmm_data_i.reset_index()

for l, s_id in tqdm(enumerate(df_sta_metadata.site_id.values),
                    total=len(df_sta_metadata.site_id.values),
                    desc='Processing sites'):
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
    # Iterate through periods:
    for per in per2process:
        time_s = timeit.default_timer()
        scec_data_filename = f'df_scec_{s_id}_T{per:.2f}s.pkl'
        df_scec_data = pd.read_pickle(os.path.join(dir_scec_extracted, scec_data_filename))
        print(f'Time for load scec pickle: {timeit.default_timer() - time_s:.2f} seconds')
        time_s = timeit.default_timer()
        for k, gmm_key in enumerate(gmm_dict):
            #add scenario id and psa gmm information
            df_scec_data_res = df_scec_data.merge(
                df_gmm_data[gmm_key][['scen_id','source_id','rupture_id','site_id']+['gmm_T%.2fs'%per]], 
                left_on=['Source_ID','Rupture_ID','Site_Name'],
                right_on=['source_id','rupture_id','site_id'])
            print('Time used for merging SCEC and metadata for site %s gmm %s: %.1f sec'%(s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()
            #add period information
            df_scec_data_res.loc[:,'period'] = per
            #scale scec psa from cm/sec^2 to g
            df_scec_data_res.loc[:,'psa'] *= 1/grav * 0.01
            #compute residuals
            df_scec_data_res.loc[:,'res'] = np.log(df_scec_data_res.psa.values) - \
                np.log(df_scec_data_res.loc[:,'gmm_T%.2fs'%per].values)
            #gmm name
            df_scec_data_res.loc[:,'gmm'] = gmm_key
            print('Time used for calculating residual for site %s gmm %s: %.1f sec'%(s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()
            #append to sql database
            if k == 0:
                df_scec_data_res[['scen_id','rup_var_id','period','psa']].to_sql(f'data_sa_scec_{s_id}',      con=db_gm_cnx, index=False, if_exists='append', method='multi', chunksize=5000)
                # df_scec_data_res[['scen_id','rup_var_id','period','psa']].to_sql('data_sa_scec', con=engine, index=False, if_exists='append', method='multi', chunksize=1000)
                print('Time used for scec to_sql (%.0f) for site %s gmm %s: %.1f sec'%(len(df_scec_data_res), s_id, gmm_key, timeit.default_timer() - time_s))
                time_s = timeit.default_timer()
            df_scec_data_res[['scen_id','rup_var_id','gmm','period','res']].to_sql(f'data_sa_res_{s_id}', con=db_gm_cnx, index=False, if_exists='append', method='multi', chunksize=5000)
            # df_scec_data_res[['scen_id','rup_var_id','gmm','period','res']].to_sql('data_sa_res', con=engine, index=False, if_exists='append', method='multi', chunksize=1000)
            print('Time used for res to_sql (%.0f) for site %s gmm %s: %.1f sec'%(len(df_scec_data_res), s_id, gmm_key, timeit.default_timer() - time_s))
            time_s = timeit.default_timer()

