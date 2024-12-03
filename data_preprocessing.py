import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime

def load_and_prepare_data(train_file, test_file, columns_to_use):
    # Load training data
    mat_train = loadmat(train_file)['train_data'][:, columns_to_use]
    mat_train[:, -1] = (mat_train[:, -1] - np.mean(mat_train[:, -1])) / np.std(mat_train[:, -1])
    
    # Create training DataFrame
    df_train = pd.DataFrame(mat_train, columns=['oniw', 'pmmw', 'oni', 'pmm', 'GPP'], 
                            index=pd.date_range('2001-01-15', '2020-12-31', freq='ME'))
    
    # Load test data
    mat_test = loadmat(test_file)['test_data'][0]
    
    return df_train, mat_test

def create_lagged_features(df, shift_idx=5):
    t1 = df.iloc[shift_idx:].copy()
    for lag in range(1, shift_idx + 1):
        for col in df.columns[:-1]:  # Exclude GPP for lag creation
            t1[f'{col}_lag_{lag}'] = df[col].shift(lag).iloc[shift_idx-lag:-lag].values
    
    # Move 'GPP' to the end
    cols = t1.columns.tolist()
    cols.remove('GPP')
    cols.append('GPP')
    t1 = t1[cols]
    return t1

def prepare_lead_data(df, lead):
    df_lead = df.iloc[:-lead].copy()
    df_lead[f'GPP_lead_{lead}'] = df['GPP'].iloc[lead:].values
    return df_lead