#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# universally modules
import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
from tqdm import tqdm

# preprocessing and transformation modules
import fasttext
import Preprocessing
from Features import buildFeatures
from Modelling import StackingModelling
from ModelSelection import ModelSelection, process_case
from Transformation import StackedTransformation, transformation

# Scikit-Learn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer

# model algorithm
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor

# evaluation modules
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# # Individual Modelling
# 
# This part is dependent of the cases and need to be done individual for each case

# ### Model Selection

# In[ ]:


# Split the data into  X and y
case_gender = {"target_type": "classification",
               "target_variable": "gender",
               "grid_search_metric": "accuracy",
               "text_features": "text_preprocessed",
               "categorial_variables": ["topic","sign"],
               "min_df_exponents" : [(1/2), (1/8)],
               "n_gram_range" : [(1,2)],
               "use_tfidf": [True],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1],
                                   'max_depth': [6],
                                   'n_estimators': [800, 2000]}),
                                  
                                  (SGDClassifier,
                                   {"loss": ["log"],
                                   "penalty": ["elasticnet"],
                                   "alpha": [0.3, 0.1, 0.01]})]}

case_topic = {"target_type": "classification",
               "target_variable": "topic",
               "grid_search_metric": "f1_weighted",
               "text_features": "text_preprocessed",
               "categorial_variables": ["gender","sign"],
               "min_df_exponents" : [(1/2), (1/8)],
               "n_gram_range" : [(1,1)],
               "use_tfidf": [True],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1, 0.3],
                                   'max_depth': [6],
                                   'n_estimators': [200, 600]}),
                                  
                                  (LinearSVC,{"penalty": ["l2"],
                                   "loss": ["squared_hinge"],
                                   "C": [0.4, 0.6, 0.8]})]}

case_age = {"target_type": "regression",
               "target_variable": "age",
               "grid_search_metric": "neg_mean_squared_error",
               "text_features": "text_preprocessed",
               "categorial_variables": ["topic","gender","sign"],
               "min_df_exponents" : [(1/2), (1/8)],
               "n_gram_range" : [(1,2)],
               "use_tfidf": [True],
               "ml_algorithms_params": [(XGBRegressor,
                                   {'learning_rate': [0.7, 1, 1.2],
                                   'max_depth': [6],
                                   'n_estimators': [400, 800]}),
                                  
                                  
                                  (LinearSVR,
                                   {"loss": ["squared_epsilon_insensitive"],
                                   "C": [0.3, 0.4, 0.8]})]}




case_sign = {"target_type": "classification",
               "target_variable": "sign",
               "grid_search_metric": "f1_weighted",
               "text_features": "text_preprocessed",
               "categorial_variables": ["gender","topic"],
               "min_df_exponents" : [(1/2), (1/8)],
               "n_gram_range" : [(1,2)],
               "use_tfidf": [True],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1, 0.3],
                                   'max_depth': [6],
                                   'n_estimators': [400, 800]}),
                                  
                                  (SGDClassifier,
                                   {"loss": ["log"],
                                   "penalty": ["elasticnet"],
                                   "alpha": [0.3, 0.1, 0.01]})]}


# In[ ]:


df_filtered = pd.read_pickle("./df_full_preprocessed.pkl")

for case in [case_gender, case_sign, case_age, case_topic]:
    df_results = process_case(transformation, case, df_filtered)
    df_results.to_pickle(f'./Model Selection/pd_df_cv_{case["target_variable"]}_{str(pd.Timestamp.now())}.pkl')

