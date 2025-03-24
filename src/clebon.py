# https://www.youtube.com/watch?v=QWGrpwO4O_k&t=1s

import librosa as lb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

seed = 18

# Séparation train et test
df = pd.read_csv('data/features_30_sec.csv')
df = df.drop(['filename', 'length'], axis=1)
df_train = df.sample(frac=0.9, random_state=seed)
df_test = df.drop(df_train.index)
# print(df_train.shape, df_test.shape)

# Séparation X et y
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
print(X_train.shape)
print(y_train.shape)
X_test=df_test.drop('label',axis=1).values
y_test=df_test['label'].values
print(X_test.shape)
print(y_test.shape)

# Création du pipeline
params_svm = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf", "sigmoid"],
    "svm__gamma": ["scale", "auto"]
}

# SVM = Support Vector Machine
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('variance', VarianceThreshold()),
    ('svm', SVC())
])

grid_svm = GridSearchCV(pipe_svm, param_grid=params_svm, cv=5, n_jobs=-1)
grid_svm.fit(X_train, y_train)
print(grid_svm.best_params_)
print(grid_svm.best_score_)
y_pred = grid_svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

joblib.dump(grid_svm, './models/pipe_svm.joblib') # Modèle sauvegardé

# # Réutilisation du modèle
# pipe_svm = joblib.load('./models/pipe_svm.joblib')
# y_pred = pipe_svm.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# XGB = Xtreme Gradient Boost
params_xgb = {
    "xgb__max_depth": [4, 10, 20],
    "xgb__booster": ["gbtree", "dart"]
}

# encodage des labels car XGB ne supporte pas les labels non-encodés
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

pipe_xgb = Pipeline([
    ('var_tresh', VarianceThreshold(threshold=0.1)),
    ('xgb', XGBClassifier(objective='multi:softmax',num_class=10,verbosity=1))
])
grid_xgb = GridSearchCV(pipe_xgb, params_xgb, scoring='accuracy', n_jobs=-1, cv=3)
grid_xgb.fit(X_train, y_train)
preds1 = grid_xgb.predict(X_test)

print(grid_xgb.best_params_)
print(grid_xgb.best_score_)
print(accuracy_score(y_test, preds1))

joblib.dump(grid_xgb, './models/pipe_xgb.joblib') # Modèle sauvegardé
