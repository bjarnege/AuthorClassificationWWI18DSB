#!/usr/bin/env python
# coding: utf-8

# In[1]:


# universally modules
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append("../src")

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


# ### Parameters

# In[11]:


# to speed up the process choose a sample size to randomly draw a sample of the whole daataset
sample_size = 10**100 

# remove all text that contain less than n chars
min_chars_per_text = 50

# which features will be used for the TF-IDF transformation
text_features = "text_preprocessed"


# # General Preprocessing
# 
# This part is independent from the cases

# ### Initialization 

# In[3]:


df = pd.read_csv("../resource/data/blogtext.csv")

# draw random sample for faster processing:
df = df.sample(sample_size)


# ### Filtering

# In[4]:


# filter for a mininmal number of letters in a tweet:
df = df[df["text"].str.count(r"[a-zA-Z]") >= min_chars_per_text]
df = df.reset_index(drop=True)


# ### Feature Engineering

# In[5]:


# append the data
features = [buildFeatures(text) for text  in tqdm(df["text"])]

# merge the features with the original dataset
df_preprocessed = df.merge(pd.DataFrame(features), left_index=True, right_index=True)


# ### Text Preprocessing

# In[6]:


# use the preprocessing  module
preprocessor = Preprocessing.Preprocessing()
df_preprocessed["text_preprocessed"] = preprocessor.ProcessMany(df_preprocessed["text"])

# predict the main language
model = fasttext.load_model('../src/data/lid.176.ftz')
df_preprocessed["main_language"] = [model.predict(text)[0][0].split("__")[-1] for text in df_preprocessed["text_preprocessed"]]

# drop unnecassary features
df_filtered = df_preprocessed[(df_preprocessed["main_language"] == "en")]                .drop(["id","text","date","main_language"], axis= 1)


# In[ ]:


# store df as pickle
df_filtered.to_pickle("./df_full_preprocessed.pkl")


# # Individual Transformation
# 
# This part is dependent of the cases and need to be done individual for each case

# ### Clustering

# In[7]:


def clustering():

    text_data_features = stacking.text_transformer.get_feature_names()
    text_data = stacking.X_train_text_transformed.toarray()

    df_text_cluster = pd.DataFrame(text_data, columns=text_data_features)

    # Die Features beschreiben die Worte im Text
    # Die Werte sind die TF*IDF-transformierten Textdaten

    numerical_data = stacking.X_train_numerical_transformed
    numerical_features = np.append(stacking.numerical_transformer.transformers_[0][1].get_feature_names(),                         stacking.X_train_numerical.columns.drop(categorial_variables))

    df_numerical_cluster = pd.DataFrame(numerical_data, columns=numerical_features)
    df_numerical_cluster


    # Die Features mit den x0 - xi Werten beschreiben die Ausprägungen die kategorialen Variablen
    # das jeweilige i beschreibt das i-te Element der im Punkt "Target Variable" definierten liste categorial_variables
    # Die verbleibenden Features (ohne xi) sind Standardskaliert, (x - \mu)/\sigma

    # THIS PART IS STILL MISSING


# # Individual Modelling
# 
# This part is dependent of the cases and need to be done individual for each case

# ### Model Selection

# In[8]:


# Split the data into  X and y
case_gender = {"target_type": "classification",
               "target_variable": "gender",
               "grid_search_metric": "accuracy",
               "text_features": "text_preprocessed",
               "categorial_variables": ["topic","sign"],
               "min_df_exponents" : [(1/4), (1/3), (1/2)],
               "n_gram_range" : [(1,1), (1,2), (2,2)],
               "use_tfidf": [True, False],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1, 1, 1.5],
                                   'max_depth': [3, 6, 9],
                                   'n_estimators': [200, 600, 1200]}),
                                  
                                  (SGDClassifier,
                                   {"loss": ["hinge", "log", "modified_huber", "squared_hinge"],
                                   "penalty": ["l2", "l1", "elasticnet"],
                                   "alpha": [0.00001, 0.001]}),
                                  
                                  (LinearSVC,{"penalty": ["l1", "l2"],
                                   "loss": ["hinge", "squared_hinge"],
                                   "C": [0.8, 1, 1.2]})]
              }


case_topic = {"target_type": "classification",
               "target_variable": "topic",
               "grid_search_metric": "f1_weighted",
               "text_features": "text_preprocessed",
               "categorial_variables": ["gender","sign"],
               "min_df_exponents" : [(1/4), (1/3), (1/2)],
               "n_gram_range" : [(1,1), (1,2), (2,2)],
               "use_tfidf": [True, False],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1, 1, 1.5],
                                   'max_depth': [3, 6, 9],
                                   'n_estimators': [200, 600, 1200]}),
                                  
                                  (SGDClassifier,
                                   {"loss": ["hinge", "log", "modified_huber", "squared_hinge"],
                                   "penalty": ["l2", "l1", "elasticnet"],
                                   "alpha": [0.01, 0.1, 1]}),
                                  
                                  (LinearSVC,{"penalty": ["l1", "l2"],
                                   "loss": ["hinge", "squared_hinge"],
                                   "C": [0.8, 1, 1.2]})]
              }



case_age = {"target_type": "regression",
               "target_variable": "age",
               "grid_search_metric": "neg_mean_squared_error",
               "text_features": "text_preprocessed",
               "categorial_variables": ["topic","gender","sign"],
               "min_df_exponents" : [(1/4), (1/3), (1/2)],
               "n_gram_range" : [(1,1), (1,2), (2,2)],
               "use_tfidf": [True, False],
               "ml_algorithms_params": [(XGBRegressor,
                                   {'learning_rate': [0.1, 1, 1.5],
                                   'max_depth': [3, 6, 9],
                                   'n_estimators': [200, 600, 1200]}),
                                  
                                  (SGDRegressor,
                                   {"loss": ["squared_loss", "huber"],
                                   "penalty": ["l2", "l1", "elasticnet"],
                                   "alpha": [0.01, 0.1, 1]}),
                                  
                                  (LinearSVR,
                                   {"loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                                   "C": [0.8, 1, 1.2]})]
              }


case_topic = {"target_type": "classification",
               "target_variable": "sign",
               "grid_search_metric": "f1_weighted",
               "text_features": "text_preprocessed",
               "categorial_variables": ["gender","topic"],
               "min_df_exponents" : [(1/4), (1/3), (1/2)],
               "n_gram_range" : [(1,1), (1,2), (2,2)],
               "use_tfidf": [True, False],
               "ml_algorithms_params": [(XGBClassifier,
                                   {'learning_rate': [0.1, 1, 1.5],
                                   'max_depth': [3, 6, 9],
                                   'n_estimators': [200, 600, 1200]}),
                                  
                                  (SGDClassifier,
                                   {"loss": ["hinge", "log", "modified_huber", "squared_hinge"],
                                   "penalty": ["l2", "l1", "elasticnet"],
                                   "alpha": [0.01, 0.1, 1]}),
                                  
                                  (LinearSVC,{"penalty": ["l1", "l2"],
                                   "loss": ["hinge", "squared_hinge"],
                                   "C": [0.8, 1, 1.2]})]
              }


# In[10]:


for case in [case_gender, case_topic, case_age, case_topic]:
    df_results = process_case(transformation, case, df_filtered)
    df_results.to_pickle(f'./pd_df_cv_{case["target_variable"]}_{str(pd.Timestamp.now())}.pkl')


# In[ ]:




