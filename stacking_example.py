#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:12:32 2017

@author: Thor
"""


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.qda import QDA
import numpy as np
from sklearn.cross_validation import KFold


def my_classifier_predictions(X_train,Y_train,X_test):
    qd = QDA()
    gb_1 = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    lr = LogisticRegression()
    ab = AdaBoostClassifier()
    ag = xg.XGBClassifier()
    
    base_models = [qd, gb_1, rf, dt, lr, ab, xg]
    
    params = [
        {},
        {'n_estimators':[150,300,500,700],'learning_rate':[.01,.1,.5],'max_leaf_nodes':[None,50,100,200]},
        {'n_estimators':[200, 700],'max_features': ['auto', 'sqrt', 'log2']},
        {'criterion':['gini','entropy'],'max_depth':[5,10,20,100]},
        {'C':[.01,.05,.1]},
        {'n_estimators':[100,200,500]},
        {}
            ]
    best_models = []
    
    for i,model in enumerate(base_models):
        CV_rfc = GridSearchCV(estimator=model, param_grid=params[i], cv= 5, scoring='roc_auc')
        CV_rfc.fit(X_train.toarray(),Y_train)
        best_models.append(CV_rfc.best_estimator_)
    
    
    vc = VotingClassifier(estimators=[
    ('qd',best_models[0]),('gb',best_models[1]),('rf',best_models[2]),('dt',best_models[3]),\
    ('lr',best_models[4]),('ab',best_models[5]),('xg',best_models[6])],\
     voting='soft', weights=[2,1,1,1.25,1.5,1,2])
        
    vc.fit(X_train.toarray(), Y_train)
    
    Y_pred_voted = vc.predict_proba(X_test.toarray())[:,1]
    
    nfolds = 5
    stacker = best_models[6]
    
    models = best_models[:6]
    
    Y_pred_stacked = stack_predict(X_train,Y_train,X_test)
    Y_pred = .7*Y_pred_voted + .3*Y_pred_stacked
    
    return Y_pred

def stack_predict(X_train, Y_train, X_test):
    X = X_train.toarray()
    y = Y_train
    T = X_test.toarray()
    folds = list(KFold(len(y), n_folds=nfolds, shuffle=True, random_state=5))
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))
    for i, clf in enumerate(models):
        S_test_i = np.zeros((T.shape[0], len(folds)))
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            # y_holdout = y[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = clf.predict(T)[:]
        S_test[:, i] = S_test_i.mean(1)
    stacker.fit(S_train, y)
    y_pred = stacker.predict_proba(S_test)[:,1]
    return y_pred


utils.generate_submission("../deliverables/test_features.txt",Y_pred)

