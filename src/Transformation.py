# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:48:53 2020

@author: Bjarne Gerdes
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class StackedTransformation:
    
    def  __init__(self, X=None, y=None, numerical_transformer=None, text_transformer=None, text_features=None):
        self.X = X
        self.y = y
        self.numerical_transformer = numerical_transformer
        self.text_transformer = text_transformer
        self.text_features = text_features
        
    def data_split(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def build_transformer(self, X=None, y=None,\
                          numerical_transformer=None, text_transformer=None, text_features=None):
        # check if variables are present
        if X == None:
            X = self.X
        if y == None:
            y = self.y
        if numerical_transformer == None:
            numerical_transformer = self.numerical_transformer
        if text_transformer == None:
            text_transformer = self.text_transformer
        if text_features == None:
            text_features = self.text_features

        # split the data
        
        self.data_split(X, y)
        
        # Create datasets for each classifier
        self.X_train_text = self.X_train[text_features]
        self.X_test_text =  self.X_test[text_features]
        
        self.X_train_numerical = self.X_train.drop(text_features, axis=1)
        self.X_test_numerical = self.X_test.drop(text_features, axis=1)

        # create transformers 
        self.numerical_transformer = numerical_transformer
        self.text_transformer = text_transformer
        
        #create transformed training batches
        self.X_train_numerical_transformed = self.numerical_transformer.fit_transform(self.X_train_numerical)
        self.X_test_numerical_transformed = self.numerical_transformer.transform(self.X_test_numerical)

        self.X_train_text_transformed = self.text_transformer.fit_transform(self.X_train_text)
        self.X_test_text_transformed = self.text_transformer.transform(self.X_test_text)

    def transform_one(self, x):
        entry = pd.DataFrame(columns=x.keys())
        entry = entry.append(x, ignore_index=True)
        
        # create datasets for each transformer
        entry_text = entry[self.text_features]
        entry_numerical = entry.drop(self.text_features, axis=1)
        
        entry_numerical_transformed = self.numerical_transformer.transform(entry_numerical)
        entry_text_transformed = self.text_transformer.transform(entry_text)
        
        return {"transformed_text": entry_text_transformed,\
               "transformed_numerical": entry_numerical_transformed}


    def transform_many(self, X):
        # create datasets for each transformer
        X_text = X[self.text_features]
        X_numerical = X.drop(self.text_features, axis=1)
        
        X_numerical_transformed = self.numerical_transformer.transform(X_numerical)
        X_text_transformed = self.text_transformer.transform(X_text)
        
        return {"transformed_text": X_text_transformed,\
               "transformed_numerical": X_numerical_transformed}
            
            
def transformation(X, y, target_variable, categorial_variables, \
                   min_df_exponent, ngram_range, text_features, use_idf):

    # use the  text transformer class to create two transformers for the textual and the numerical model
    text_transformer = TfidfVectorizer(ngram_range=ngram_range, min_df=int(len(X)**(min_df_exponent)), use_idf=use_idf)
    numerical_transformer = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categorial_variables)\
                                                           , remainder=StandardScaler())

    stacking = StackedTransformation(X, y, numerical_transformer, text_transformer, text_features)
    stacking.build_transformer()
    
    return stacking