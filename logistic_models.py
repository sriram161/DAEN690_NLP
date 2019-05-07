import pandas as pd
import numpy as np

# Models
from sklearn.linear_model import LogisticRegression

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
data_path = r"C:/Users/notme/Documents/Development/DAEN690_NPL/data_prep/final_data/"
lit_df = pd.read_csv(data_path+"lit.csv")
pos_df = pd.read_csv(data_path+"pos.csv")
neg_df = pd.read_csv(data_path+"neg.csv")
unc_df = pd.read_csv(data_path+"unc.csv")
thl_df = pd.read_csv(data_path+"thala.csv")
bag_df = pd.read_csv(data_path+"bagfw.csv")
wnt_df = pd.read_csv(data_path+"wordnet.csv")
tag_df = pd.read_csv(data_path+"tag.csv")

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

# STEP-2: Create a random split of test and train.
 
k_fold = 10

# STEP-2: best parameters selection for the model Logistic regression.
# Config: Settings and parameters
lr = LogisticRegression(solver='lbfgs', multi_class='ovr')
lr_sentiment = LogisticRegression(solver='lbfgs', multi_class='auto')
lr_parameters = {'C': [1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4]}

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

# Create grid seaerch for cross validation.
gs_relavence_lr = GridSearchCV(lr, lr_parameters, cv=k_fold, n_jobs=-1)
gs_sentiment_lr = GridSearchCV(lr_sentiment, lr_parameters, cv=k_fold, n_jobs=-1)

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

lr_lit = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_lit = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_lit.fit(lit_x_train, lit_y_train)
lr_sentiment_lit.fit(lit_x_train, lit_z_train)

lr_pos = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_pos = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_pos.fit(pos_x_train, pos_y_train)
lr_sentiment_pos.fit(pos_x_train, pos_z_train)

lr_neg = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_neg = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_neg.fit(neg_x_train, neg_y_train)
lr_sentiment_neg.fit(neg_x_train, neg_z_train)

lr_unc = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_unc = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_unc.fit(unc_x_train, unc_y_train)
lr_sentiment_unc.fit(unc_x_train, unc_z_train)

lr_thl = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_thl = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_thl.fit(thl_x_train, thl_y_train)
lr_sentiment_thl.fit(thl_x_train, thl_z_train)

y_predict_relavence_lit = lr_lit.predict(lit_x_test)
y_predict_sentiment_lit = lr_sentiment_lit.predict(lit_x_test)
y_predict_relavence_pos = lr_pos.predict(pos_x_test)
y_predict_sentiment_pos = lr_sentiment_pos.predict(pos_x_test)
y_predict_relavence_neg = lr_neg.predict(neg_x_test)
y_predict_sentiment_neg = lr_sentiment_neg.predict(neg_x_test)
y_predict_relavence_unc = lr_unc.predict(unc_x_test)
y_predict_sentiment_unc = lr_sentiment_unc.predict(unc_x_test)

y_predict_relavence_thl = lr_thl.predict(thl_x_test)
y_predict_sentiment_thl = lr_sentiment_thl.predict(thl_x_test)

# STEP-2.7: plot confusion matrix.
def plot_confusion_matrix(test:pd.DataFrame, predict:pd.DataFrame, name:str):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test, predict)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title(f'Relavent or Not relavent Confusion Matrix - Test {name} Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

plot_confusion_matrix(lit_y_test, y_predict_relavence_lit, 'lit')
plot_confusion_matrix(pos_y_test, y_predict_relavence_pos, 'pos')
plot_confusion_matrix(neg_y_test, y_predict_relavence_neg, 'neg')
plot_confusion_matrix(unc_y_test, y_predict_relavence_unc, 'unc')
plot_confusion_matrix(thl_y_test, y_predict_relavence_thl, 'thl')

# STEP-2.8: precison_recall_fscore_suport metrics
print(classification_report(lit_y_test, y_predict_relavence_lit))
print(classification_report(pos_y_test, y_predict_relavence_pos))
print(classification_report(neg_y_test, y_predict_relavence_neg))
print(classification_report(unc_y_test, y_predict_relavence_unc))
print(classification_report(thl_y_test, y_predict_relavence_thl))

# STEP-2.9: TODO: Plot AUC.


def plot_roc_auc(test: pd.DataFrame, predict: pd.DataFrame, name: str):
    bin_form_test = [1 if i =='relevant' else 0 for i in test]
    bin_form_predict = [1 if i == 'relevant' else 0 for i in predict]
    fpr, tpr, _ = roc_curve(bin_form_test, bin_form_predict, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic example {name}')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_auc(lit_y_test, y_predict_relavence_lit, 'lit')
plot_roc_auc(pos_y_test, y_predict_relavence_pos, 'pos')
plot_roc_auc(neg_y_test, y_predict_relavence_neg, 'neg')
plot_roc_auc(unc_y_test, y_predict_relavence_unc, 'unc')
plot_roc_auc(thl_y_test, y_predict_relavence_thl, 'thl')


# Important features.
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
mic = mutual_info_classif(x_bag_df,  y_relavence_bag_df)
s = pd.DataFrame()
s['att'] = x_bag_df.columns
s['mic'] = mic
s['chi2'] = chi2(x_bag_df, y_relavence_bag_df)[0]
features = s.sort_values('chi2', ascending=False)['att'][:]
list(features)

lr_thl = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
lr_sentiment_thl = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr_thl.fit(thl_x_train[list(features)], thl_y_train)
lr_sentiment_thl.fit(thl_x_train[list(features)], thl_z_train)
y_predict_relavence_thl = lr_thl.predict(thl_x_test[list(features)])
y_predict_sentiment_thl = lr_sentiment_thl.predict(thl_x_test[list(features)])
plot_confusion_matrix(thl_y_test, y_predict_relavence_thl, 'thl')


# Less sparse logistic

bag_x_train, bag_x_test, bag_y_train, bag_y_test, bag_z_train, bag_z_test = train_test_split(
    x_bag_df, y_relavence_bag_df, y_sentiment_bag_df, shuffle=True, test_size=0.25, random_state=1)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
mic = mutual_info_classif(bag_x_train,  bag_y_train)
s = pd.DataFrame()
s['att'] = bag_x_train.columns
s['mic'] = mic
s['chi2'] = chi2(bag_x_train, bag_y_train)[0]
features = s.sort_values('chi2', ascending=False)['att'][:]

import matplotlib.pyplot as plt
rf200 = LogisticRegression(C=0.1, solver='lbfgs', multi_class='ovr')
Auc =[]
No_of_Attributes =[]
for n in range(100, bag_x_train.shape[1], 500):
    cols_sel_mic=s.sort_values('chi2', ascending=False)['att'][:n]
    rf200.fit(bag_x_train[cols_sel_mic], bag_y_train)
    probs_rf200 = rf200.predict_proba(bag_x_test[cols_sel_mic])
    bin_form_test = [1 if i == 'relevant' else 0 for i in bag_y_test]
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(
        bin_form_test, probs_rf200[:,1],  pos_label=1)
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    Auc.append(auc_rf200)
    No_of_Attributes.append(n)
    plt.title('Logit')
    plt.plot(fpr_rf200, tpr_rf200,label =" {} Number of attributes".format(n))
    plt.legend()


aucperf = pd.DataFrame({'No_of_Attributes': No_of_Attributes, 'Auc': Auc})
aucperf.plot.scatter(x='No_of_Attributes', y='Auc',c='DarkBlue', title='Logit')
plt.show()
