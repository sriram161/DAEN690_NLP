from app.helpers.plotter import plot_confusion_matrix
from app.helpers.model_exec import model_exec
from app.helpers.selection import metric_data_prep
from app.helpers.selection import get_mic_chi2_s_df
from app.helpers.model_exec import model_eval
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as acc
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.ion()
import os
path_ = os.getcwd()

# STEP-1: Count vectorizer data import.
data_path = path_ + r"/data_prep/final_data/strategy/"
lit_df = pd.read_csv(data_path+"lit.csv")
pos_df = pd.read_csv(data_path+"pos.csv")
neg_df = pd.read_csv(data_path+"neg.csv")
unc_df = pd.read_csv(data_path+"unc.csv")
thl_df = pd.read_csv(data_path+"thala.csv")
bag_df = pd.read_csv(data_path+"bagfw.csv")
wnt_df = pd.read_csv(data_path+"wordnet.csv")
tag_df = pd.read_csv(data_path+"tag.csv")

# Create data frames for feature engineering.
x_lit_df = lit_df[lit_df.columns[1:-2]]
y_relavence_lit_df = lit_df['Relevance']
y_sentiment_lit_df = lit_df['Sentiment']
x_pos_df = pos_df[pos_df.columns[1:-2]]
y_relavence_pos_df = pos_df['Relevance']
y_sentiment_pos_df = pos_df['Sentiment']
x_neg_df = neg_df[neg_df.columns[1:-2]]
y_relavence_neg_df = neg_df['Relevance']
y_sentiment_neg_df = neg_df['Sentiment']
x_unc_df = unc_df[unc_df.columns[1:-2]]
y_relavence_unc_df = unc_df['Relevance']
y_sentiment_unc_df = unc_df['Sentiment']
x_thl_df = thl_df[thl_df.columns[1:-2]]
y_relavence_thl_df = thl_df['Relevance']
y_sentiment_thl_df = thl_df['Sentiment']
x_bag_df = bag_df[bag_df.columns[1:-2]]
y_relavence_bag_df = bag_df['Relevance']
y_sentiment_bag_df = bag_df['Sentiment']
x_wnt_df = wnt_df[wnt_df.columns[1:-2]]
y_relavence_wnt_df = wnt_df['Relevance']
y_sentiment_wnt_df = wnt_df['Sentiment']
x_tag_df = tag_df[tag_df.columns[1:-2]]
y_relavence_tag_df = tag_df['Relevance']
y_sentiment_tag_df = tag_df['Sentiment']

# STEP2: Caclutate chi for all 8 datasets
lit_x_train, lit_x_test, lit_y_train, lit_y_test, lit_z_train, lit_z_test = train_test_split(
    x_lit_df, y_relavence_lit_df, y_sentiment_lit_df, shuffle=True, test_size=0.25, random_state=1)
pos_x_train, pos_x_test, pos_y_train, pos_y_test, pos_z_train, pos_z_test = train_test_split(
    x_pos_df, y_relavence_pos_df, y_sentiment_pos_df, shuffle=True, test_size=0.25, random_state=1)
neg_x_train, neg_x_test, neg_y_train, neg_y_test, neg_z_train, neg_z_test = train_test_split(
    x_neg_df, y_relavence_neg_df, y_sentiment_neg_df, shuffle=True, test_size=0.25, random_state=1)
unc_x_train, unc_x_test, unc_y_train, unc_y_test, unc_z_train, unc_z_test = train_test_split(
    x_unc_df, y_relavence_unc_df, y_sentiment_unc_df, shuffle=True, test_size=0.25, random_state=1)
thl_x_train, thl_x_test, thl_y_train, thl_y_test, thl_z_train, thl_z_test = train_test_split(
    x_thl_df, y_relavence_thl_df, y_sentiment_thl_df, shuffle=True, test_size=0.25, random_state=1)
bag_x_train, bag_x_test, bag_y_train, bag_y_test, bag_z_train, bag_z_test = train_test_split(
    x_bag_df, y_relavence_bag_df, y_sentiment_bag_df, shuffle=True, test_size=0.25, random_state=1)
wnt_x_train, wnt_x_test, wnt_y_train, wnt_y_test, wnt_z_train, wnt_z_test = train_test_split(
    x_wnt_df, y_relavence_wnt_df, y_sentiment_wnt_df, shuffle=True, test_size=0.25, random_state=1)
tag_x_train, tag_x_test, tag_y_train, tag_y_test, tag_z_train, tag_z_test = train_test_split(
    x_tag_df, y_relavence_tag_df, y_sentiment_tag_df, shuffle=True, test_size=0.25, random_state=1)

# Compute statistical evaluation mic = information gain.
# Compute statistical evaluation chi2 = chi square test.
lit_s = get_mic_chi2_s_df(lit_x_train, lit_y_train)
pos_s = get_mic_chi2_s_df(pos_x_train, pos_y_train)
neg_s = get_mic_chi2_s_df(neg_x_train, neg_y_train)
unc_s = get_mic_chi2_s_df(unc_x_train, unc_y_train)
bag_s = get_mic_chi2_s_df(bag_x_train, bag_y_train)
wnt_s = get_mic_chi2_s_df(wnt_x_train, wnt_y_train)
tag_s = get_mic_chi2_s_df(tag_x_train, tag_y_train)
thl_s = get_mic_chi2_s_df(thl_x_train, thl_y_train)

# STEP3: Run the logistic model
# aucperf = pd.DataFrame({'No_of_Attributes': No_of_Attributes, 'Auc': Auc})
# aucperf.plot.scatter(x='No_of_Attributes', y='Auc', c='DarkBlue', title=title+str((max(Auc_dict.values())))
# plt.show()
# STEP4: number of attributes vs AUC s for 8 datasets.
print('lit',lit_x_train.shape)
print('pos',pos_x_train.shape)
print('neg',neg_x_train.shape)
print('unc',unc_x_train.shape)
print('bag',bag_x_train.shape)
print('wnt',wnt_x_train.shape)
print('tag',tag_x_train.shape)
print('thl',thl_x_train.shape)
lit_start = 2
pos_start = 3
neg_start = 7
unc_start = 2
bag_start = 200
wnt_start = 3
tag_start = 37
thl_start = 2
skip = 10

model_exec(GaussianNB, lit_x_train, lit_x_test, lit_y_train, lit_y_test, lit_start, int(np.ceil(np.linspace(lit_start, lit_x_train.shape[1], skip)[2] - np.linspace(lit_start, lit_x_train.shape[1], skip)[1])), 'lit',
           lit_s, 'chi2')
model_exec(GaussianNB, pos_x_train, pos_x_test, pos_y_train, pos_y_test, pos_start, int(np.ceil(np.linspace(pos_start, pos_x_train.shape[1], skip)[2] - np.linspace(pos_start, pos_x_train.shape[1], skip)[1])), 'pos',
           pos_s, 'chi2')
model_exec(GaussianNB, neg_x_train, neg_x_test, neg_y_train, neg_y_test, neg_start, int(np.ceil(np.linspace(neg_start, neg_x_train.shape[1], skip)[2] - np.linspace(neg_start, neg_x_train.shape[1], skip)[1])), 'neg',
           neg_s, 'chi2')
model_exec(GaussianNB, unc_x_train, unc_x_test, unc_y_train, unc_y_test, unc_start, int(np.ceil(np.linspace(unc_start, unc_x_train.shape[1], skip)[2] - np.linspace(unc_start, unc_x_train.shape[1], skip)[1])), 'unc',
           unc_s, 'chi2')
model_exec(GaussianNB, bag_x_train, bag_x_test, bag_y_train, bag_y_test, bag_start, int(np.ceil(np.linspace(bag_start, bag_x_train.shape[1], skip)[2] - np.linspace(bag_start, bag_x_train.shape[1], skip)[1])), 'bag',
           bag_s, 'chi2')
model_exec(GaussianNB, wnt_x_train, wnt_x_test, wnt_y_train, wnt_y_test, wnt_start, int(np.ceil(np.linspace(wnt_start, wnt_x_train.shape[1], skip)[2] - np.linspace(wnt_start, wnt_x_train.shape[1], skip)[1])), 'wnt',
           wnt_s, 'chi2')
model_exec(GaussianNB, thl_x_train, thl_x_test, thl_y_train, thl_y_test, thl_start, int(np.ceil(np.linspace(thl_start, thl_x_train.shape[1], skip)[2] - np.linspace(thl_start, thl_x_train.shape[1], skip)[1])), 'thl',
           thl_s, 'chi2')
model_exec(GaussianNB, tag_x_train, tag_x_test, tag_y_train, tag_y_test, tag_start, int(np.ceil(np.linspace(tag_start, tag_x_train.shape[1], skip)[2] - np.linspace(tag_start, tag_x_train.shape[1], skip)[1])), 'tag',
           tag_s, 'chi2')

# MIC
model_exec(GaussianNB, lit_x_train, lit_x_test, lit_y_train, lit_y_test, lit_start, int(np.ceil(np.linspace(lit_start, lit_x_train.shape[1], skip)[2] - np.linspace(lit_start, lit_x_train.shape[1], skip)[1])), 'lit',
           lit_s, 'mic')
model_exec(GaussianNB, pos_x_train, pos_x_test, pos_y_train, pos_y_test, pos_start, int(np.ceil(np.linspace(pos_start, pos_x_train.shape[1], skip)[2] - np.linspace(pos_start, pos_x_train.shape[1], skip)[1])), 'pos',
           pos_s, 'mic')
model_exec(GaussianNB, neg_x_train, neg_x_test, neg_y_train, neg_y_test, neg_start, int(np.ceil(np.linspace(neg_start, neg_x_train.shape[1], skip)[2] - np.linspace(neg_start, neg_x_train.shape[1], skip)[1])), 'neg',
           neg_s, 'mic')
model_exec(GaussianNB, unc_x_train, unc_x_test, unc_y_train, unc_y_test, unc_start, int(np.ceil(np.linspace(unc_start, unc_x_train.shape[1], skip)[2] - np.linspace(unc_start, unc_x_train.shape[1], skip)[1])), 'unc',
           unc_s, 'mic')
model_exec(GaussianNB, bag_x_train, bag_x_test, bag_y_train, bag_y_test, bag_start, int(np.ceil(np.linspace(bag_start, bag_x_train.shape[1], skip)[2] - np.linspace(bag_start, bag_x_train.shape[1], skip)[1])), 'bag',
           bag_s, 'mic')
model_exec(GaussianNB, wnt_x_train, wnt_x_test, wnt_y_train, wnt_y_test, wnt_start, int(np.ceil(np.linspace(wnt_start, wnt_x_train.shape[1], skip)[2] - np.linspace(wnt_start, wnt_x_train.shape[1], skip)[1])), 'wnt',
           wnt_s, 'mic')
model_exec(GaussianNB, thl_x_train, thl_x_test, thl_y_train, thl_y_test, thl_start, int(np.ceil(np.linspace(thl_start, thl_x_train.shape[1], skip)[2] - np.linspace(thl_start, thl_x_train.shape[1], skip)[1])), 'thl',
           thl_s, 'mic')
model_exec(GaussianNB, tag_x_train, tag_x_test, tag_y_train, tag_y_test, tag_start, int(np.ceil(np.linspace(tag_start, tag_x_train.shape[1], skip)[2] - np.linspace(tag_start, tag_x_train.shape[1], skip)[1])), 'tag',
           tag_s, 'mic')

# for chi2 data after feateure engineering
lit_sel_df = metric_data_prep(x_lit_df, lit_s, n=12, flag='chi2')
pos_sel_df = metric_data_prep(x_pos_df, pos_s, n=23, flag='chi2')
neg_sel_df = metric_data_prep(x_neg_df, neg_s, n=31, flag='chi2')
unc_sel_df = metric_data_prep(x_unc_df, unc_s, n=8, flag='chi2')
bag_sel_df = metric_data_prep(x_bag_df, bag_s, n=1802, flag='chi2')
wnt_sel_df = metric_data_prep(x_wnt_df, wnt_s, n=15, flag='chi2')
thl_sel_df = metric_data_prep(x_thl_df, thl_s, n=20, flag='chi2')
tag_sel_df = metric_data_prep(x_tag_df, tag_s, n=295, flag='chi2')

df_chi_eng = pd.concat([lit_sel_df, tag_sel_df, thl_sel_df, wnt_sel_df,
                        pos_sel_df, neg_sel_df, unc_sel_df, bag_sel_df], axis=1, sort=False)

chi_x_train, chi_x_test, chi_y_train, chi_y_test, chi_z_train, chi_z_test = train_test_split(
    df_chi_eng, y_relavence_lit_df, y_sentiment_lit_df, shuffle=True, test_size=0.25, random_state=1)

chi_s = get_mic_chi2_s_df(chi_x_train, chi_y_train)


model_exec(GaussianNB, chi_x_train, chi_x_test, chi_y_train, chi_y_test,
           100, 234, 'mic',
           chi_s, 'mic')

# for 100
chi_sel_df = metric_data_prep(df_chi_eng, chi_s, n=100, flag='chi2')
model_eval(GaussianNB, chi_sel_df, y_relavence_lit_df, title='chi2')

# MIC 
# for chi2 data after feateure engineering
lit_mic_sel_df = metric_data_prep(x_lit_df, lit_s, n=12, flag='mic')
pos_mic_sel_df = metric_data_prep(x_pos_df, pos_s, n=31, flag='mic')
neg_mic_sel_df = metric_data_prep(x_neg_df, neg_s, n=71, flag='mic')
unc_mic_sel_df = metric_data_prep(x_unc_df, unc_s, n=12, flag='mic')
bag_mic_sel_df = metric_data_prep(x_bag_df, bag_s, n=734, flag='mic')
wnt_mic_sel_df = metric_data_prep(x_wnt_df, wnt_s, n=21, flag='mic')
thl_mic_sel_df = metric_data_prep(x_thl_df, thl_s, n=17, flag='mic')
tag_mic_sel_df = metric_data_prep(x_tag_df, tag_s, n=80, flag='mic')

df_mic_eng = pd.concat([lit_mic_sel_df, tag_mic_sel_df, thl_mic_sel_df, wnt_mic_sel_df,
                        pos_mic_sel_df, neg_mic_sel_df, unc_mic_sel_df, bag_mic_sel_df], axis=1, sort=False)

mic_x_train, mic_x_test, mic_y_train, mic_y_test, mic_z_train, mic_z_test = train_test_split(
    df_mic_eng, y_relavence_lit_df, y_sentiment_lit_df, shuffle=True, test_size=0.25, random_state=1)

mic_s = get_mic_chi2_s_df(mic_x_train, mic_y_train)

model_exec(GaussianNB, mic_x_train, mic_x_test, mic_y_train, mic_y_test, 
100, 97, 'mic',
mic_s, 'mic')

model_eval(GaussianNB, df_mic_eng, y_relavence_lit_df,
           title='mic')

mic_sel_df = metric_data_prep(df_mic_eng, mic_s, n=100, flag='mic')
x = model_eval(GaussianNB, mic_sel_df, y_relavence_lit_df,
           title='mic')
x.to_csv('nb_strategy_mic.csv')

x_sel_df = pd.concat([x_lit_df, x_tag_df, x_thl_df, x_wnt_df, x_pos_df, x_neg_df, x_unc_df, x_bag_df], axis=1, sort=False)

sel_x_train, sel_x_test, sel_y_train, sel_y_test, sel_z_train, sel_z_test = train_test_split(
    x_sel_df, y_relavence_tag_df, y_sentiment_tag_df, shuffle=True, test_size=0.25, random_state=1)
