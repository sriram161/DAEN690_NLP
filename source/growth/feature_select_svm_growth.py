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
from sklearn.svm import SVC
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
data_path = path_ + r"/data_prep/final_data/growth/"
lit_df = pd.read_csv(data_path+"lit.csv")
pos_df = pd.read_csv(data_path+"pos.csv")
neg_df = pd.read_csv(data_path+"neg.csv")
unc_df = pd.read_csv(data_path+"unc.csv")
thl_df = pd.read_csv(data_path+"thala.csv")
bag_df = pd.read_csv(data_path+"bagfw.csv")
wnt_df = pd.read_csv(data_path+"wordnet.csv")
tag_df = pd.read_csv(data_path+"tag.csv")

# Create data frames for feature engineering.
x_lit_df = lit_df[lit_df.columns[1:-3]]
y_relavence_lit_df = lit_df['Relevance']
y_sentiment_lit_df = lit_df['Sentiment']
x_pos_df = pos_df[pos_df.columns[1:-3]]
y_relavence_pos_df = pos_df['Relevance']
y_sentiment_pos_df = pos_df['Sentiment']
x_neg_df = neg_df[neg_df.columns[1:-3]]
y_relavence_neg_df = neg_df['Relevance']
y_sentiment_neg_df = neg_df['Sentiment']
x_unc_df = unc_df[unc_df.columns[1:-3]]
y_relavence_unc_df = unc_df['Relevance']
y_sentiment_unc_df = unc_df['Sentiment']
x_thl_df = thl_df[thl_df.columns[2:-2]]
y_relavence_thl_df = thl_df['Relevance']
y_sentiment_thl_df = thl_df['Sentiment']
x_bag_df = bag_df[bag_df.columns[2:-2]]
y_relavence_bag_df = bag_df['Relevance']
y_sentiment_bag_df = bag_df['Sentiment']
x_wnt_df = wnt_df[wnt_df.columns[1:-2]]
y_relavence_wnt_df = wnt_df['Relevance']
y_sentiment_wnt_df = wnt_df['Sentiment']
x_tag_df = tag_df[tag_df.columns[1:-3]]
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
x_sel_df = pd.concat([x_lit_df, x_tag_df, x_thl_df, x_wnt_df, x_pos_df, x_neg_df, x_unc_df, x_bag_df], axis=1, sort=False)

sel_x_train, sel_x_test, sel_y_train, sel_y_test, sel_z_train, sel_z_test = train_test_split(
    x_sel_df, y_relavence_tag_df, y_sentiment_tag_df, shuffle=True, test_size=0.25, random_state=1)

sel = SelectFromModel(SVC(kernel='linear', probability=True))
sel.fit(sel_x_train, sel_y_train)
sel.get_support()
selected_feat = sel_x_train.columns[(sel.get_support())]
len(selected_feat)

x = model_eval(SVC, x_sel_df[selected_feat], y_relavence_lit_df,
           title='sfm', kernel='linear', probability=True)
x.to_csv('svm_growth_sfm.csv')