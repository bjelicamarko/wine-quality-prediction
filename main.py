import numpy as np
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

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from util import *

import sys

import time
import warnings
warnings.filterwarnings('ignore')

def make_correlation_matrix(df):
    plt.figure(figsize=(12, 12))
    sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
    plt.show()

def train_test_validation_with_scaling(features, target, scaler = None):
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

    if scaler is not None:
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    return xtrain, xtest, ytrain, ytest

def classification(xtrain, xtest, ytrain, ytest, classification_models, list_of_parameters):
    for i in range(len(classification_models)):
        print(f'{classification_models[i]} : ')

        gs = GridSearchCV(classification_models[i], list_of_parameters[i], n_jobs=-1, cv=5, scoring='f1_micro')
        gs.fit(xtrain, ytrain)

        print("TRAINING")
        print("Best estimator from gridsearch: {}".format(gs.best_estimator_))
        print("Best parameters from gridsearch: {}".format(gs.best_params_))
        print("Best f1 score={}".format(gs.best_score_))

        print("TESTING")
        print("F Measure: " + 
            str(metrics.f1_score(ytest, gs.best_estimator_.predict(xtest),  average='micro')))
        print(metrics.classification_report(ytest,
                                            gs.best_estimator_.predict(xtest)))

        print()

def regression(xtrain, xtest, ytrain, ytest, regression_models, list_of_parameters):
    for i in range(len(regression_models)):
        print(f'{regression_models[i]} : ')

        gs = GridSearchCV(regression_models[i], list_of_parameters[i], n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        gs.fit(xtrain, ytrain)

        print("TRAINING")
        print("Best estimator from gridsearch: {}".format(gs.best_estimator_))
        print("Best parameters from gridsearch: {}".format(gs.best_params_))
        print("RMSE score={}".format(gs.best_score_))

        print("TESTING")
        print("RMSE: " + 
            str(metrics.mean_squared_error(ytest, gs.best_estimator_.predict(xtest), squared=True)))

        print()

def create_ann_model(Optimizer_Trial, Neurons_Trial):
    classifier = Sequential()

    classifier.add(Dense(units=Neurons_Trial, input_dim=13, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=Neurons_Trial, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=Optimizer_Trial, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier
    
def neural_network(xtrain, xtest, ytrain, ytest, parameters):
    classifier_model = KerasClassifier(create_ann_model, verbose=0)

    gs = GridSearchCV(classifier_model, parameters, n_jobs=-1, cv=5, scoring='f1_micro')

    start_time = time.time()

    gs.fit(xtrain, ytrain, verbose=1)

    end_time = time.time()
    print("Total Time Taken: ", round((end_time -start_time)/60), 'Minutes')

    print("TRAINING")
    print("Best estimator from gridsearch: {}".format(gs.best_estimator_))
    # gs.best_estimator_.save("my_model")
    print("Best parameters from gridsearch: {}".format(gs.best_params_))
    print("Best f1 score={}".format(gs.best_score_))

    print("TESTING")
    print("F Measure: " + 
        str(metrics.f1_score(ytest, gs.best_estimator_.predict(xtest),  average='micro')))
    print(metrics.classification_report(ytest, gs.best_estimator_.predict(xtest)))

    print()

def main(argv):
    df = pd.read_csv(argv[1])
    features = df.drop(['quality'], axis=1)
    target = df['quality']
    
    scaler = None
    if argv[2] == 'stan':
        scaler = StandardScaler()
    elif argv[2] == 'norm':
        scaler = MinMaxScaler()
   

    xtrain, xtest, ytrain, ytest = train_test_validation_with_scaling(features, target, scaler)

    if 'reg' in argv:
        regression_models = [
            GradientBoostingRegressor(), 
            LinearRegression(), 
            Ridge()
        ]
        list_of_parameters = [
            {'learning_rate': [0.05, 0.1, 1, 10], 'max_depth': [1, 3, 5, 7, 9], 'n_estimators': [50, 150, 200, 500]},
            {},
            {'alpha': [1, 2, 3, 4, 5, 10, 15, 20, 50]}
        ]
        regression(xtrain, xtest, ytrain, ytest, regression_models, list_of_parameters)
  
    if 'cls' in argv:
        classification_models = [
            SVC(), 
            RandomForestClassifier(), 
            GaussianNB()
        ]
        list_of_parameters = [
            {'kernel': ['rbf'], 'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'degree': [1, 2, 3],
            'gamma': [0.1, 1.0, 10.0, 'scale', 'auto']}, 
            {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800]}, 
            {} 
        ]
        classification(xtrain, xtest, ytrain, ytest, classification_models, list_of_parameters)

    if 'ann' in argv:
        parameters = {
            'batch_size': [10,20,30],
            'epochs': [10,20],
            'Optimizer_Trial': ['adam', 'rmsprop'],
            'Neurons_Trial': [5,10]
        }
        neural_network(xtrain, xtest, ytrain, ytest, parameters)

if __name__ == "__main__":
    main(sys.argv)
