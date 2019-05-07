import pandas as pd
import numpy as np

# Models
from sklearn.ensemble import RandomForestClassifier

# Best Parameters search and test on k-fold cross validation for random sampling.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

# TODO: Evaluatation metrics plots.
# TODO: Plot the evalutaion results of the model.

# STEP-1: Count vectorizer data import.
lit_df = pd.read_csv("lit.csv")
pos_df = pd.read_csv("pos.csv")
neg_df = pd.read_csv("neg.csv")
unc_df = pd.read_csv("unc.csv")
thl_df = pd.read_csv("thala.csv")

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


# STEP-2: Create a random split of test and train.

k_fold = 10

# STEP-2: best parameters selection for the model Logistic regression.
# Config: Settings and parameters
# NOTE: This search method use maximum depth to purifi until leaf have < min_sample_split.
lr = RandomForestClassifier()
lr_sentiment = RandomForestClassifier()
lr_parameters = {'n_estimators': list(range(10, 1000, 10)),
                 'criterion': ['gini', 'entropy'],
                 'min_samples_split': [2, 10, 100]}

# STEP-2.1: lit data.
# Create grid seaerch for cross validation.
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

# Train grid of models.
gs_relavence_lr_fit = gs_relavence_lr.fit(x_lit_df, y_relavence_lit_df)
gs_sentiment_lr_fit = gs_sentiment_lr.fit(x_lit_df, y_sentiment_lit_df)

# matrix data frame.
lr_lit_relavence_metrics_df = pd.DataFrame(gs_relavence_lr_fit.cv_results_)
lr_lit_sentiment_metrics_df = pd.DataFrame(gs_sentiment_lr_fit.cv_results_)

# STEP-2.2: pos data.
# Create grid seaerch for cross validation.
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

# Train grid of models.
gs_relavence_lr_fit = gs_relavence_lr.fit(x_pos_df, y_relavence_pos_df)
gs_sentiment_lr_fit = gs_sentiment_lr.fit(x_pos_df, y_sentiment_pos_df)

# matrix data frame.
lr_pos_relavence_metrics_df = pd.DataFrame(gs_relavence_lr_fit.cv_results_)
lr_pos_sentiment_metrics_df = pd.DataFrame(gs_sentiment_lr_fit.cv_results_)

# STEP-2.3: neg data.
# Create grid seaerch for cross validation.
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

# Train grid of models.
gs_relavence_lr_fit = gs_relavence_lr.fit(x_neg_df, y_relavence_neg_df)
gs_sentiment_lr_fit = gs_sentiment_lr.fit(x_neg_df, y_sentiment_neg_df)

# matrix data frame.
lr_neg_relavence_metrics_df = pd.DataFrame(gs_relavence_lr_fit.cv_results_)
lr_neg_sentiment_metrics_df = pd.DataFrame(gs_sentiment_lr_fit.cv_results_)

# STEP-2.4: unc data.
# Create grid seaerch for cross validation.
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

# Train grid of models.
gs_relavence_lr_fit = gs_relavence_lr.fit(x_unc_df, y_relavence_unc_df)
gs_sentiment_lr_fit = gs_sentiment_lr.fit(x_unc_df, y_sentiment_unc_df)

# matrix data frame.
lr_unc_relavence_metrics_df = pd.DataFrame(gs_relavence_lr_fit.cv_results_)
lr_unc_sentiment_metrics_df = pd.DataFrame(gs_sentiment_lr_fit.cv_results_)

#STEP-2.5: Thala data
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(
    lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

# Train grid of models.
gs_relavence_lr_fit = gs_relavence_lr.fit(x_thl_df, y_relavence_thl_df)
gs_sentiment_lr_fit = gs_sentiment_lr.fit(x_thl_df, y_sentiment_thl_df)

# matrix data frame.
lr_thl_relavence_metrics_df = pd.DataFrame(gs_relavence_lr_fit.cv_results_)
lr_thl_sentiment_metrics_df = pd.DataFrame(gs_sentiment_lr_fit.cv_results_)

# STEP-2.5: Plot evaluation metric to choose parameters.
lr_unc_relavence_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_unc_sentiment_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_pos_relavence_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_pos_sentiment_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_neg_relavence_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_neg_sentiment_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_lit_relavence_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_lit_sentiment_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_thl_relavence_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]
lr_thl_sentiment_metrics_df[['mean_fit_time', 'mean_score_time', 'mean_test_score', 'params', 'rank_test_score']]

# QUESTION: How to choose solvers on models? is it dependent on data? 
# Using rank_test_score select best test model.

##### 'mean_fit_time', 'mean_score_time',  'rank_test_score'
# STEP-2.6: Train model with best choosen parameters.
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


# lr_lit = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_sentiment_lit = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_lit.fit(lit_x_train, lit_y_train)
# lr_sentiment_lit.fit(lit_x_train, lit_z_train)

# lr_pos = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_sentiment_pos = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_pos.fit(pos_x_train, pos_y_train)
# lr_sentiment_pos.fit(pos_x_train, pos_z_train)

# lr_neg = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_sentiment_neg = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_neg.fit(neg_x_train, neg_y_train)
# lr_sentiment_neg.fit(neg_x_train, neg_z_train)

# lr_unc = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_sentiment_unc = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_unc.fit(unc_x_train, unc_y_train)
# lr_sentiment_unc.fit(unc_x_train, unc_z_train)

# lr_thl = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_sentiment_thl = RandomForestClassifier(n_estimators=, criterion=, min_samples_split=)
# lr_thl.fit(thl_x_train, thl_y_train)
# lr_sentiment_thl.fit(thl_x_train, thl_z_train)

# y_predict_relavence_lit = lr_lit.predict(lit_x_test)
# y_predict_sentiment_lit = lr_sentiment_lit.predict(lit_x_test)
# y_predict_relavence_pos = lr_pos.predict(pos_x_test)
# y_predict_sentiment_pos = lr_sentiment_pos.predict(pos_x_test)
# y_predict_relavence_neg = lr_neg.predict(neg_x_test)
# y_predict_sentiment_neg = lr_sentiment_neg.predict(neg_x_test)
# y_predict_relavence_unc = lr_unc.predict(unc_x_test)
# y_predict_sentiment_unc = lr_sentiment_unc.predict(unc_x_test)
# y_predict_relavence_thl = lr_thl.predict(thl_x_test)
# y_predict_sentiment_thl = lr_sentiment_thl.predict(thl_x_test)

# # STEP-2.7: plot confusion matrix.
# def plot_confusion_matrix(test:pd.DataFrame, predict:pd.DataFrame, name:str):
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(test, predict)
#     plt.clf()
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
#     classNames = ['Negative', 'Positive']
#     plt.title(f'Relavent or Not relavent Confusion Matrix - Test {name} Data')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     tick_marks = np.arange(len(classNames))
#     plt.xticks(tick_marks, classNames, rotation=45)
#     plt.yticks(tick_marks, classNames)
#     s = [['TN', 'FP'], ['FN', 'TP']]
#     for i in range(2):
#         for j in range(2):
#             plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
#     plt.show()

# plot_confusion_matrix(lit_y_test, y_predict_relavence_lit, 'lit')
# plot_confusion_matrix(pos_y_test, y_predict_relavence_pos, 'pos')
# plot_confusion_matrix(neg_y_test, y_predict_relavence_neg, 'neg')
# plot_confusion_matrix(unc_y_test, y_predict_relavence_unc, 'unc')
# plot_confusion_matrix(thl_y_test, y_predict_relavence_thl, 'thl')

# # STEP-2.8: precison_recall_fscore_suport metrics
# print(classification_report(lit_y_test, y_predict_relavence_lit))
# print(classification_report(pos_y_test, y_predict_relavence_pos))
# print(classification_report(neg_y_test, y_predict_relavence_neg))
# print(classification_report(unc_y_test, y_predict_relavence_unc))
# print(classification_report(thl_y_test, y_predict_relavence_thl))

# # STEP-2.9: Plot AUC.
# def plot_roc_auc(test: pd.DataFrame, predict: pd.DataFrame, name: str):
#     bin_form_test = [1 if i =='relevant' else 0 for i in test]
#     bin_form_predict = [1 if i == 'relevant' else 0 for i in predict]
#     fpr, tpr, _ = roc_curve(bin_form_test, bin_form_predict, pos_label=1)
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver operating characteristic example {name}')
#     plt.legend(loc="lower right")
#     plt.show()

# plot_roc_auc(lit_y_test, y_predict_relavence_lit, 'lit')
# plot_roc_auc(pos_y_test, y_predict_relavence_pos, 'pos')
# plot_roc_auc(neg_y_test, y_predict_relavence_neg, 'neg')
# plot_roc_auc(unc_y_test, y_predict_relavence_unc, 'unc')
# plot_roc_auc(thl_y_test, y_predict_relavence_thl, 'thl')