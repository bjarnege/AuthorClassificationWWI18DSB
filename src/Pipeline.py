# universally modules
import sys
import numpy as np
import pandas as pd

# preprocessing and transformation modules
import fasttext
import Preprocessing
from Features import buildFeatures
from Modelling import StackingModelling
from ModelSelection import ModelSelection, process_case
from Transformation import StackedTransformation, transformation

# model algorithm
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor


# Bestimmen der besten Lernalgorithmen und Optimierungsparameter für die jeweilige Zielvariable
# und bauen der daraus resultierenden Pipeline

class BuildPipeline:
    
    def __init__(self, df_eval_results_variable, df_preprocessed, algo_type):
        self.df_eval_results_variable = df_eval_results_variable
        self.target_variable = df_eval_results_variable["target_variable"].values[0]
        self.categorial_variables = df_eval_results_variable["categorial_variables"].values[0]
        self.X = df_preprocessed.drop(self.target_variable, axis=1)
        self.y = df_preprocessed[self.target_variable]

        self.algo_type = algo_type
        
    def best_model_and_params(self):
        # get best textual_model
        self.best_text_model = self.df_eval_results_variable[self.df_eval_results_variable["CV_text_best_score"] ==\
                                              self.df_eval_results_variable["CV_text_best_score"].max()]

        self.best_text_algo = self.best_text_model["ml_algorithms_params"].values[0][0]
        self.best_text_params = self.best_text_model["CV_text_best_params"].values[0]

        # get best numerical model
        self.best_numerical_model = self.df_eval_results_variable[self.df_eval_results_variable["CV_numerical_best_score"]\
                                                  == self.df_eval_results_variable["CV_numerical_best_score"].max()]

        self.best_numerical_algo = self.best_numerical_model["ml_algorithms_params"].values[0][0]
        self.best_numerical_params = self.best_numerical_model["CV_text_best_params"].values[0]
    
    def build_transformation(self):
        min_df_exponent = self.best_text_model["min_df_exponents"].values[0]
        ngram_range = self.best_text_model["n_gram_range"].values[0]
        text_features = self.best_text_model["text_features"].values[0]
        use_idf = self.best_text_model["use_tfidf"].values[0]
        self.transformation = transformation(self.X, self.y, self.target_variable, self.categorial_variables,\
                                             min_df_exponent, ngram_range, text_features, use_idf)
    
    def build_model(self):
        self.modelling = StackingModelling(self.best_numerical_algo, self.best_numerical_params,\
                                     self.best_text_algo, self.best_text_params, self.transformation)
        
        self.modelling.fit()
        print(self.modelling.create_report(self.transformation.X_test, self.transformation.y_test, self.algo_type))
    
    
    def fit(self):
        self.best_model_and_params()
        self.build_transformation()
        self.build_model()
        
    def predict(self, X):
        return self.modelling.weighted_prediction(X, algo_type=self.algo_type)
# Erstellen der Transformation für text und numerischen Datensatz