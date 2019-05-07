from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as acc
from app.helpers.plotter import plot_confusion_matrix

def model_exec(f, x_train, x_test, y_train, y_test, start, step, title, s, flag, **kwargs):
    model = f(**kwargs)
    Auc = []
    Auc_dict = {}
    No_of_Attributes = []
    plt.figure()
    for n in range(start, x_train.shape[1], step):
        cols_sel = s.sort_values(flag, ascending=False)['att'][:n]
        model.fit(x_train[cols_sel], y_train)
        probs_model = model.predict_proba(x_test[cols_sel])
        bin_form_test = [1 if i == 'relevant' else 0 for i in y_test]
        fpr_model, tpr_model, thresholds_model = roc_curve(
            bin_form_test, probs_model[:, 1],  pos_label=1)
        auc_model = auc(fpr_model, tpr_model)
        Auc.append(auc_model)
        Auc_dict[n] = Auc
        No_of_Attributes.append(n)
        plt.title(title)
        plt.plot(fpr_model, tpr_model, label=" {} Number of attributes".format(
            n) + r"(AUC="+str(round((Auc[-1]), 3))+r")")
        plt.legend()
    plt.show()

def model_eval(model_f, x_df, y_df, title, **kwargs):
    chi_x_train, chi_x_test, chi_y_train, chi_y_test = tts(
        x_df, y_df, shuffle=True, test_size=0.25, random_state=1)
    lr_chi = model_f(**kwargs)
    lr_chi.fit(chi_x_train, chi_y_train)
    y_predict_relavence_chi = lr_chi.predict(chi_x_test)
    print(classification_report(chi_y_test, y_predict_relavence_chi))
    plot_confusion_matrix(chi_y_test, y_predict_relavence_chi, title)
    print(acc(chi_y_test, y_predict_relavence_chi))
    probs_model = lr_chi.predict_proba(chi_x_test)
    bin_form_test = [1 if i == 'relevant' else 0 for i in chi_y_test]
    fpr_model, tpr_model, thresholds_model = roc_curve(
            bin_form_test, probs_model[:, 1],  pos_label=1)
    print("AUC: ", auc(fpr_model, tpr_model))
    return pd.DataFrame({'fpr': fpr_model,'tpr': tpr_model, 'threshold': thresholds_model})
