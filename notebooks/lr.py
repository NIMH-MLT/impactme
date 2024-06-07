import pandas as pd
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression

def fit_lr(X_trn, y_trn, X_tst, y_tst, max_iter=1000, seed=1, C=1.0):
    """Given C, fit a logistic regression model to training data and return predictions and scores on testing data
    
    Parameters
    ----------
    X_trn: {array-like, sparse matrix} of shape (n_samples, n_features)
        Predictors for training the model

    y_trn: array-like of shape (n_samples,)
        Response variable for training

    X_tst: {array-like, sparse matrix} of shape (n_samples, n_features)
        Predictors for testing the model

    y_tst: array-like of shape (n_samples,)
        Response variables for checking the test predictions

    C: num, default=1.0
        Inverse regularization parameter

    max_iter: int, default=1000
        Max iterations during logistic regression

    seed: int, default=1
        Random seed during the logreg fit

    Returns
    -------
    pd.Dataframe
        'y': y values
        'y_type': either true (test values), predicted labels, or probabilities
        'C': C-value used
    """
    clf = LogisticRegression(
        class_weight='balanced', 
        max_iter=max_iter, 
        random_state=seed,
        C=C
    )
    clf.fit(X_trn, y_trn)

    y_prob = clf.predict_proba(X_tst)
    y_prob = [i[1] for i in y_prob]

    df = pd.DataFrame({
        'y_true': y_tst, 
        'y_pred': clf.predict(X_tst), 
        'y_prob': y_prob
    })
    
    df['C'] = C

    del clf
    return df