import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

def metric_data_prep(data:pd.DataFrame, s:pd.DataFrame, n:int, flag:str) -> pd.DataFrame:
    cols_sel = s.sort_values(flag, ascending=False)['att'][:n]
    return data[cols_sel]

def get_mic_chi2_s_df(x_train:pd.DataFrame, y_train:pd.DataFrame)-> pd.DataFrame:
    chi_s = pd.DataFrame()
    chi_s['mic'] = mutual_info_classif(x_train, y_train)
    chi_s['chi2'] = chi2(x_train, y_train)[0]
    chi_s['att'] = x_train.columns
    return chi_s
