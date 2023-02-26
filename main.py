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
from sklearn.model_selection import GridSearchCV

from util import *

import warnings
warnings.filterwarnings('ignore')

def make_correlation_matrix(df):
    plt.figure(figsize=(12, 12))
    sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
    plt.show()

def train_test_validation_with_scaling(features, target, scaler):
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    return xtrain, xtest, ytrain, ytest

def cross_validation_with_scaling():
    pass

def classification(xtrain, xtest, ytrain, ytest, classification_models, list_of_parameters):
    for i in range(len(classification_models)):
        print(f'{classification_models[i]} : ')

        gs = GridSearchCV(classification_models[i], list_of_parameters[i], n_jobs=-1, cv=5, scoring='f1_micro')
        gs.fit(xtrain, ytrain)

        print("TRAINING")
        print("Best estimator from gridsearch: {}".format(gs.best_estimator_))
        print("Best parameters from gridsearch: {}".format(gs.best_params_))
        print("CV score={}".format(gs.best_score_))

        print("TESTING")
        print("F Measure: " + 
            str(metrics.f1_score(ytest, gs.best_estimator_.predict(xtest),  average='micro')))
        print(metrics.classification_report(ytest,
                                            gs.best_estimator_.predict(xtest)))

        print()

def regression(xtrain, xtest, ytrain, ytest, regression_models):
    for i in range(len(regression_models)):
        print(f'{regression_models[i]} : ')

        regression_models[i].fit(xtrain, ytrain)

        print("RMSE: " + 
            str(metrics.mean_squared_error(ytest, regression_models[i].predict(xtest),  squared=True)))
        print()

def main():
    df = pd.read_csv('dataset.csv')
    features = df.drop(['quality'], axis=1)
    target = df['quality']
    
    # standardization
    xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, StandardScaler())

    # normalization
    # xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, MinMaxScaler())

    # regression_models = [
    #     GradientBoostingRegressor(), 
    #     LinearRegression(), 
    #     Ridge()
    # ]
    # list_of_parameters = [
    #     {},
    #     {},
    #     {}
    # ]
    # regression(xtrain, xtest, ytrain, ytest, regression_models)
  
    classification_models = [
        SVC(), 
        RandomForestClassifier(), 
        GaussianNB()
    ]
    list_of_parameters = [
        {'kernel': ['rbf'], 'C': [1.0, 2.0, 3.0], 'degree': [1, 2, 3]}, 
        {'n_estimators': [100, 200, 300]}, 
        {} 
    ]
    classification(xtrain, xtest, ytrain, ytest, classification_models, list_of_parameters)

if __name__ == "__main__":
    main()
