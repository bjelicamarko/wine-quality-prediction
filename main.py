import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from util import *

def train_test_validation_with_scaling(features, target, scaler):
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    return xtrain, xtest, ytrain, ytest

def cross_validation_with_scaling():
    pass

def main():
    df = pd.read_csv('prepared_dataset.csv')
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']
  
    # standardization
    xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, StandardScaler())

    # normalization
    #xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, MinMaxScaler())

    models = [SVC(kernel='rbf'), GradientBoostingRegressor(), LinearRegression()]
    for i in range(len(models)):
        models[i].fit(xtrain, ytrain)
        print(f'{models[i]} : ')
        print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
        print('Validation Accuracy : ', metrics.roc_auc_score(
            ytest, models[i].predict(xtest)))
        print()
  
    # print(metrics.classification_report(ytest, svm.predict(xtest)))

if __name__ == "__main__":
    main()
