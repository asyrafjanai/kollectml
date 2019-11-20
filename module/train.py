import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

cache_dir = os.getcwd()[:-6]

def split_df(rs=1, test_size=0.25):

    print('Preprocessing data!')
    df_path = os.path.join(cache_dir, 'storage', 'data.xlsx')
    df = pd.read_excel(df_path)
    print(df.head())
    X = df.iloc[:, 1:-1]
    y = df[['delinquet_flg']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)

    return X_train, X_test, y_train, y_test

def preprocess_data():
    X_train, X_test, y_train, y_test = split_df()

    # normalize data here

    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    print('Training model!')
    clf = XGBClassifier()
    clf.fit(X_train, y_train.values.ravel())
    model_path = os.path.join(cache_dir, 'storage', 'model.sav')
    # clf.save_model(model_path)
    # filename = 'finalized_model.sav'
    joblib.dump(clf, model_path)
    print(f'Model saved at {model_path}')


if __name__ == '__main__':
    train_model()
