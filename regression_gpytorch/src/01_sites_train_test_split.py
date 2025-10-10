import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    # output directory for the site lists
    out_dir = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/sites_train_test_split"
    ### Load the database file
    rup_var_per_eq_file = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/rup_var_per_eq.json"
    with open(rup_var_per_eq_file, 'r') as f:
        rup_var_per_eq = json.load(f)

    fn_metadata = "/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/gm_metadata_expanded.csv"
    df_metadata = pd.read_csv(fn_metadata)

    df_metadata = df_metadata.loc[~np.isin(df_metadata.source_id, [1, 13, 41, 188]),:]
    df_metadata = df_metadata.loc[df_metadata.eq_id.astype(str).isin(list(rup_var_per_eq.keys())), :]
    df_metadata.reset_index(drop=True, inplace=True)

    ### Define the sites to process
    site_list = np.sort(df_metadata.site_id.unique())
    site_list_train, site_list_test = train_test_split(site_list, test_size=0.2, random_state=85)
    # save to csv for later use
    site_list_train_df = pd.DataFrame(site_list_train, columns=['site_id'])
    site_list_test_df = pd.DataFrame(site_list_test, columns=['site_id'])
    site_list_train_df.to_csv(os.path.join(out_dir, 'site_list_train.csv'), index=False)
    site_list_test_df.to_csv(os.path.join(out_dir, 'site_list_test.csv'), index=False)

    # Create a site metadata dataframe
    src_site_metadata_file = "/resnick/groups/enceladus/glavrent/Scalable_GPs/Raw_files/scec/study_22.12_sites.csv"
    df_site_metadata = pd.read_csv(src_site_metadata_file)
    df_site_metadata = df_site_metadata.merge(
        df_metadata[['site_id', 'site_x', 'site_y', 'utm_zone']].drop_duplicates(),
        on='site_id', how='left')
    df_site_metadata.to_csv(os.path.join(out_dir, 'site_metadata.csv'), index=False)