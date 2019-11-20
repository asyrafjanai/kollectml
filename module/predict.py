import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import *

cache_dir = os.path.join(os.getcwd()[:-6], 'storage')

def make_prediction(x):
    """
    Make batch transformation
    :return:
    csv file
    """
    # read model
    model_path = os.path.join(cache_dir, 'model.sav')
    clf = joblib.load(model_path)
    prediction = clf.predict(x)
    return prediction

if __name__ == '__main__':
    # get data to predict
    pred_path = os.path.join(cache_dir, 'test_prediction.csv')
    pred_df = pd.read_csv(pred_path)
    X_pred = pred_df.iloc[:, 1:]
    y_prediction = make_prediction(X_pred)
    print(X_pred.shape)
    print(y_prediction.shape)

    prediction = pred_df.iloc[:, :1]
    prediction['prediction'] = y_prediction
    print(prediction.shape)
    print(prediction.head())

    prediction_path = os.path.join(cache_dir, 'prediction_df.csv')
    prediction.to_csv(prediction_path, index=False)