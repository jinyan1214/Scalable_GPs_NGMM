import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from training_utils.utils import read_hdf5_with_site_id_rrup
import pandas as pd
import h5py

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
    parser.add_argument('--training_data_fp', type=str, 
                        # default="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/formatted_data/training_4179_eqs/training_sites_training_eqs_2.00_ASK14_4179_eqs_all_var_per_eq.h5",
                        default="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/formatted_data/testing_4179_eqs/training_sites_testing_eqs_2.00_ASK14_4179_eqs_all_var_per_eq.h5",
                        help='Path to the training data file')
    parser.add_argument('--testing_data_fp', type=str, 
                        # default="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/formatted_data/training_4179_eqs/testing_sites_training_eqs_2.00_ASK14_4179_eqs_all_var_per_eq.h5",
                        default="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/formatted_data/testing_4179_eqs/testing_sites_testing_eqs_2.00_ASK14_4179_eqs_all_var_per_eq.h5",
                        help='Path to the testing data file')
    
    args = parser.parse_args()
    training_data_fp = args.training_data_fp
    testing_data_fp = args.testing_data_fp
    
    print("Formatting training data to compute mean and std residuals per site and earthquake...")
    X_train, Y_train, site_id_train, rrup_train = read_hdf5_with_site_id_rrup(training_data_fp)
    train_data = pd.DataFrame(X_train, columns=['eq_var_id','source_cst_x', 'source_cst_y', 
            'source_cst_z', 'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
            'site_x', 'site_y', 'site_z'])
    train_data['res'] = Y_train
    train_data['Rrup'] = rrup_train
    train_data['site_id'] = site_id_train
    train_data['eq_id'] = train_data['eq_var_id'].apply(lambda x: int(x//1000))
    print(f"Total training records: {len(train_data)}")
    mean_train = train_data.groupby(['site_id', 'eq_id'])['res'].mean().reset_index()
    std_train = train_data.groupby(['site_id', 'eq_id'])['res'].std().reset_index()
    print("Computed mean and std residuals per site and training earthquake.")
    train_data_unique = train_data.drop_duplicates(subset=['site_id', 'eq_id'], keep='first').reset_index(drop=True)
    train_data_unique = train_data_unique.drop(columns=['res'])
    mean_train = mean_train.merge(train_data_unique, on=['site_id', 'eq_id'], how='left')
    std_train = std_train.merge(train_data_unique, on=['site_id', 'eq_id'], how='left')
    mean_train_fp = training_data_fp.replace('.h5', '_mean.h5')
    std_train_fp = training_data_fp.replace('.h5', '_std.h5')
    save_to_hdf5(mean_train, mean_train_fp)
    save_to_hdf5(std_train, std_train_fp)

    print("Formatting testing data to compute mean and std residuals per site and earthquake...")
    X_test, Y_test, site_id_test, rrup_test = read_hdf5_with_site_id_rrup(testing_data_fp)
    test_data = pd.DataFrame(X_test, columns=['eq_var_id','source_cst_x', 'source_cst_y', 
            'source_cst_z', 'source_mpt_x', 'source_mpt_y', 'source_mpt_z',
            'site_x', 'site_y', 'site_z'])
    test_data['res'] = Y_test
    test_data['Rrup'] = rrup_test
    test_data['site_id'] = site_id_test
    test_data['eq_id'] = test_data['eq_var_id'].apply(lambda x: int(x//1000))
    print(f"Total testing records: {len(test_data)}")
    mean_test = test_data.groupby(['site_id', 'eq_id'])['res'].mean().reset_index()
    std_test = test_data.groupby(['site_id', 'eq_id'])['res'].std().reset_index()
    test_data_unique = test_data.drop_duplicates(subset=['site_id', 'eq_id'], keep='first').reset_index(drop=True)
    test_data_unique = test_data_unique.drop(columns=['res'])
    mean_test = mean_test.merge(test_data_unique, on=['site_id', 'eq_id'], how='left')
    std_test = std_test.merge(test_data_unique, on=['site_id', 'eq_id'], how='left')
    mean_test_fp = testing_data_fp.replace('.h5', '_mean.h5')
    std_test_fp = testing_data_fp.replace('.h5', '_std.h5')
    save_to_hdf5(mean_test, mean_test_fp)
    save_to_hdf5(std_test, std_test_fp)
    print("Formatting completed.")