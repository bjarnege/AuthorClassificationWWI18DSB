# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:59:50 2020

@author: Bjarne Gerdes
"""
import numpy as np
import pandas as pd

class StackingModelling:
    
    
    def __init__(self,  numerical_model, numerical_model_params,\
                text_model, text_model_params, stacked_transformation_instance,\
                weights=(0.5, 0.5)):
        
        # Initialize numerical model
        self.numerical_model = numerical_model(**numerical_model_params)
        
        # Initialize text model
        self.text_model = text_model(**text_model_params)
        
        # internalize transformation class
        self.stacked_transformation = stacked_transformation_instance
        
        # model weights
        self.weights = weights
        
        
    def fit(self):
        # train numerical model
        self.numerical_model.fit(self.stacked_transformation.X_train_numerical_transformed,\
                             self.stacked_transformation.y_train)
                             
        print("Numerical model finished!")
        
        # train textual model
        self.text_model.fit(self.stacked_transformation.X_train_text_transformed,\
                             self.stacked_transformation.y_train)
                             
        print("Text model finished!")

        
    def predict_text(self, X):
        return self.text_model.predict(X)
    
    
    def predict_numerical(self, X):
        return self.numerical_model.predict(X)

    
    def optimize_weights(self, X, y, algo_type="classification"):
        X_transformed = self.stacked_transformation.transform_many(X)

        if algo_type == "classification":
            class_label_dict = dict([(k,v) for v,k in enumerate(self.numerical_model.classes_)])
            y_class_index = [class_label_dict[value] for value in y]
            
            # predict each row of X for each model
            y_pred_num = self.numerical_model.predict_proba(X_transformed["transformed_numerical"])
            y_pred_text = self.text_model.predict_proba(X_transformed["transformed_text"])
            
            # absolute loss for each model
            self.loss_numerical = sum([sum(np.delete(y_p, y_t)) for y_p, y_t in zip(y_pred_num,y_class_index)])
            self.loss_text = sum([sum(np.delete(y_p, y_t)) for y_p, y_t in zip(y_pred_text,y_class_index)])
            
            # optimize the weights based on their contribution to the loss
            # bewusst text und numerical vertauscht, damit die Gegenwahrscheinlichkeit verwendet wird.
            self.weights = np.array((self.loss_numerical, self.loss_text))/(self.loss_numerical + self.loss_text)
            

        if algo_type == "regression":
            # absolute loss for each model
            self.loss_numerical = np.absolute(self.numerical_model.predict(X_transformed["transformed_numerical"]) - y).sum()
            self.loss_text = np.absolute(self.text_model.predict(X_transformed["transformed_text"]) - y).sum()
            
            # optimize the weights based on their contribution to the loss
            # bewusst text und numerical vertauscht, damit die Gegenwahrscheinlichkeit verwendet wird.
            self.weights = np.array((self.loss_numerical, self.loss_text))/(self.loss_numerical + self.loss_text)
            
            
        print(f"""Weights have been optimized:
                Textual model weight: {self.weights[0]}
                Numerical model weight: {self.weights[1]}""")

    def weighted_prediction(self, X, weights=None, algo_type="classification"):
        if weights == None:
            weights = self.weights
        
        # check if one or more transactions become processed
        try:
            if type(X) == dict:
                X_transformed = self.stacked_transformation.transform_one(X)

            else:
                X_transformed = self.stacked_transformation.transform_many(X)
        except TypeError:
            print("Check the data type of the input data")

        if algo_type == "classification":
            predictions = (self.text_model.predict_proba(X_transformed["transformed_text"]),\
                            self.numerical_model.predict_proba(X_transformed["transformed_numerical"]))
            
            index = lambda x: np.argmax(x)
            
            self.weighted_predictions = (predictions[0]*weights[0] + predictions[1]*weights[1])
            classes_name = np.array([self.text_model.classes_[index(prediction)] for prediction in self.weighted_predictions])
            
            # get name of the classes
            return classes_name
        
        if algo_type == "regression":
            predictions = (self.text_model.predict(X_transformed["transformed_text"]),\
                            self.numerical_model.predict(X_transformed["transformed_numerical"]))
                        
            self.weighted_predictions = (predictions[0]*weights[0] + predictions[1]*weights[1])
            
            return self.weighted_predictions
        
        else:
            print("target variable type not supported")
            
    def create_report(self, X, y, algo_type="classification"):
        # fit weight optimization
        self.optimize_weights(X,y, algo_type)
        
        #get loss for bagged models:
        if algo_type == "classification":
            class_label_dict = dict([(k,v) for v,k in enumerate(self.numerical_model.classes_)])
            y_class_index = [class_label_dict[value] for value in y]
            
            # get loss for unoptimized weighting
            self.weighted_prediction(X, (0.5,0.5), algo_type="classification")
            y_pred = self.weighted_predictions
            loss_weights = sum([sum(np.delete(y_p, y_t)) for y_p, y_t in zip(y_pred,y_class_index)])
            
            # get loss for optimized weights
            self.weighted_prediction(X, algo_type="classification")
            y_pred = self.weighted_predictions
            loss_weights_optimized = sum([sum(np.delete(y_p, y_t)) for y_p, y_t in zip(y_pred,y_class_index)])
        
        if algo_type == "regression":
            # get loss for unoptimized weighting
            self.weighted_prediction(X, (0.5,0.5), algo_type="regression")
            y_pred = self.weighted_predictions
            loss_weights = np.absolute(y_pred - y).sum()
  
            # get loss for optimized weighting
            self.weighted_prediction(X, algo_type="regression")
            y_pred = self.weighted_predictions
            loss_weights_optimized = np.absolute(y_pred - y).sum()
            
        self.report = pd.Series()
        self.report["Absolute loss textual model"] = self.loss_text
        self.report["Absolute loss numerical model"] = self.loss_numerical
        self.report["Absolute loss equally weighted model"] = loss_weights
        self.report["Absolute loss optimized weights model"] = loss_weights_optimized

        return self.report