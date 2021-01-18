# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:22:51 2020

@author: Bjarne Gerdes
"""

import pandas as pd
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV



class ModelSelection:
    
    def __init__(self, numerical_model, numerical_model_params_dict,\
                 text_model, text_model_params_dict, StackedTransformationInstance):
        
        self.numerical_model = numerical_model
        self.numerical_model_params_dict = numerical_model_params_dict
        
        self.text_model = text_model
        self.text_model_params_dict = text_model_params_dict
        
        self.stacking = StackedTransformationInstance
        self.results = dict()
        
    def GridSearch(self, model, params_dict, X_train, y_train,\
                   X_test, y_test, scoring="accuracy", cv=3):
        
        gs_ = GridSearchCV(model(), scoring=scoring,\
                            param_grid=params_dict, n_jobs=-1, cv=cv, verbose=4)
            
            
        # Use early stopping if model is xgb
        if model == XGBRegressor or  model == XGBClassifier:
            fit_params = {'eval_set': [(X_test, y_test)],
                          'eval_metric': "error",
                          'early_stopping_rounds': 20,
                          'verbose': False}
            
            if len(set(y_train)) > 2 and not model == XGBRegressor:
                fit_params["eval_metric"] = "merror"
                
            gs_.fit(X_train, y_train, **fit_params)

        
        else:
            gs_.fit(X_train, y_train)
        
        return {"model": model,
               "params_dict": params_dict,
               "scoring":scoring,
               "cv": cv,
               "grid_search_instance": gs_}
            
    def fit(self, scoring):
        # fit pipeline for numerical model:
        print("Starting Grid search for numerical model ...")
        
        cv_numerical = self.GridSearch(self.numerical_model, self.numerical_model_params_dict,\
                       self.stacking.X_train_numerical_transformed, self.stacking.y_train,\
                       self.stacking.X_test_numerical_transformed, self.stacking.y_test, scoring)
        
        self.results["CV_numerical"] = cv_numerical
        
        # fit pipeline for text model:
        print("Starting Grid search for text model ...")

        
        cv_text = self.GridSearch(self.text_model, self.text_model_params_dict,\
                       self.stacking.X_train_text_transformed, self.stacking.y_train,\
                       self.stacking.X_test_text_transformed, self.stacking.y_test, scoring)
        
        self.results["CV_text"] = cv_text
        
        return self.results
    
def get_cartesian(case_dict):
    case_basic = {"target_type": case_dict["target_type"],
               "target_variable": case_dict["target_variable"],
               "grid_search_metric": case_dict["grid_search_metric"],
               "text_features": case_dict["text_features"],
               "categorial_variables": case_dict["categorial_variables"]}
    
    product_variables = ["min_df_exponents", "n_gram_range", "use_tfidf", "ml_algorithms_params"]
    variations = product(*[case_dict[var] for var in product_variables])
        
        
    return [{**case_basic, **dict(zip(product_variables, elem))} for elem in variations]

def process_versions(transformation, versions_list, X ,y):
    results = []

    for version in tqdm(versions_list):
        print(f'Processing target variable {version["target_variable"]}')
        if version["target_type"] == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        stacking = transformation(X, y, version["target_variable"], version["categorial_variables"], \
                                version["min_df_exponents"], version["n_gram_range"],\
                                version["text_features"], version["use_tfidf"])

        ms = ModelSelection(version["ml_algorithms_params"][0], version["ml_algorithms_params"][1],\
                            version["ml_algorithms_params"][0], version["ml_algorithms_params"][1],\
                            stacking)
        
        cv_results = ms.fit(version["grid_search_metric"])
        cv_results = {**cv_results, **version}                            
                            
        results.append(cv_results)
        

    return results

def clean_results(results):
    df = pd.DataFrame(results)
    
    # extract the best parameter together with their errors
    df["CV_numerical_best_params"] = df["CV_numerical"].apply(lambda x: x["grid_search_instance"].best_params_)
    df["CV_numerical_best_score"] = df["CV_numerical"].apply(lambda x: x["grid_search_instance"].best_score_)
    df["CV_text_best_params"] = df["CV_text"].apply(lambda x: x["grid_search_instance"].best_params_)
    df["CV_text_best_score"] = df["CV_text"].apply(lambda x: x["grid_search_instance"].best_score_)
    
    return df

def process_case(transformation, case, df):
    X,y = df.drop(case["target_variable"], axis=1), df[case["target_variable"]]
    versions = get_cartesian(case)
    results = process_versions(transformation, versions, X, y)
    return clean_results(results)