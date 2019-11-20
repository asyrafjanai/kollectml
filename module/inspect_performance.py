import os
import joblib
import pandas as pd
import numpy as np
from module import train, predict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import *
import matplotlib.pyplot as plt

cache_dir = os.path.join(os.getcwd()[:-6], 'storage')

def inspect_model(model, X_train, y_train, X_test, y_test):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X_train, y_train.values.ravel(), cv=kfold)
    print('5-fold accuracy: ', accuracy)
    print('Mean accuracy: ', accuracy.mean(), '\n')

    y_predictions = model.predict(X_test)
    y_predprob_test = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_predictions)

    # ratio = ((test_df.click==1).sum()/test_df.shape[0]).round(2)
    # ratio = ((train_set.click==1).sum()/train_set.shape[0]).round(2)

    p, r, _ = precision_recall_curve(y_test, y_predprob_test)
    roc_auc = roc_auc_score(y_test, y_predprob_test)
    pr_auc = auc(r, p)
    fpr, tpr, threshold = roc_curve(y_test, y_predprob_test)

    print('accuracy: {}'.format(accuracy))
    print('roc_auc: {}'.format(roc_auc))
    print('pr_auc: {}\n'.format(pr_auc))

    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(y_test, y_predictions), index=['true_no', 'true_yes'],
                         columns=['pred_no', 'pred_yes']))

if __name__ == '__main__':
    model = joblib.load(os.path.join(cache_dir, 'model.sav'))
    X_train, X_test, y_train, y_test = train.preprocess_data()
    inspect_model(model, X_train, y_train, X_test, y_test)
