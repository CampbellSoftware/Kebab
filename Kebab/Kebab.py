import numpy as np
from sklearn.ensemble import *
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

class Bagging:
    @staticmethod
    def MLM(X_fit, y_fit, X, n=10):
        model = BaggingRegressor(estimator=MLPRegressor(), n_estimators=n)
        model.fit(X_fit, y_fit)
        return model.predict(np.array(X).reshape(-1, 1))

    @staticmethod
    def LinearRegressor(X_fit, y_fit, X, n=10):
        model = BaggingRegressor(estimator=LinearRegression(), n_estimators=n)
        model.fit(X_fit, y_fit)
        return model.predict(np.array(X).reshape(-1, 1))
    @staticmethod
    def Custom(X_fit, y_fit, X, estimator, n=10):
        model = BaggingRegressor(estimator=estimator, n_estimators=n)
        model.fit(X_fit, y_fit)
        return model.predict(np.array(X).reshape(-1,1))
    @staticmethod
    def CustomModel(estimator, n=10):
        model = BaggingRegressor(estimator=estimator, n_estimators=n)
        return model
        

class Sequential:
    def __init__(self, *args):
        self.models = [(f'model_{i}', model) for i, model in enumerate(args)]

    def Vote(self, X_fit, y_fit, X):
        model = VotingRegressor(estimators=self.models)
        model.fit(X_fit, y_fit)
        return model.predict(np.array(X).reshape(-1, 1))


