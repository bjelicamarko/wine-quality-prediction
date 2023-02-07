import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge

from util import *

import warnings
warnings.filterwarnings('once')

def train_test_validation_with_scaling(features, target, scaler):
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    return xtrain, xtest, ytrain, ytest

def cross_validation_with_scaling():
    pass

def classification(xtrain, xtest, ytrain, ytest, classification_models):
    for i in range(len(classification_models)):
        classification_models[i].fit(xtrain, ytrain)
        print(f'{classification_models[i]} : ')
        print("F Measure: " + 
            str(metrics.f1_score(ytest, classification_models[i].predict(xtest),  average='weighted')))
        print()

def regression(xtrain, xtest, ytrain, ytest, regression_models):
    for i in range(len(regression_models)):
        regression_models[i].fit(xtrain, ytrain)
        print(f'{regression_models[i]} : ')
        print("RMSE: " + 
            str(metrics.mean_squared_error(ytest, regression_models[i].predict(xtest),  squared=True)))
        print()

def main():
    df = pd.read_csv('prepared_dataset.csv')
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['quality']
    
    # standardization
    xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, StandardScaler())

    # normalization
    #xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, MinMaxScaler())

    regression_models = [GradientBoostingRegressor(), LinearRegression(), Ridge()]
    regression(xtrain, xtest, ytrain, ytest, regression_models)
  
    classification_models = [SVC(kernel='rbf'), RandomForestClassifier(), GaussianNB()]
    classification(xtrain, xtest, ytrain, ytest, classification_models)

if __name__ == "__main__":
    main()
