import pandas as pd
# universally modules
import sys
from Features import buildFeatures
from Pipeline import BuildPipeline
from Preprocessing import Preprocessing

# model algorithm
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor


class RequestMapper:
    
    def __init__(self, pipelines):
        # storing and unpacking the pipelines
        self.pipelines = pipelines
        self.pipeline_mapping = {"age": pipelines.values[0],
                                 "gender": pipelines.values[1],
                                 "sign": pipelines.values[2],
                                 "topic": pipelines.values[3]}
        
        # import other modules
        self.preprocessor = Preprocessing()
    
    def predict_text(self, text, target_variable):
        # preprocess text
        text_preprocessed = self.preprocessor.ProcessOne(text)
        
        # transform text to tfidf sparse matrix
        tfidf_transformed_text = self.pipeline_mapping[target_variable].\
                                transformation.text_transformer.transform({"text_preprocessed":text_preprocessed})
        return self.pipeline_mapping[target_variable].modelling.predict_text(tfidf_transformed_text)[0]
    
    def predict_numerical(self, target_variable, text, age=None, sign=None, gender=None, topic=None):
        # Build dataframe for transformer
        columns = self.pipeline_mapping[target_variable].transformation.X_test_numerical.columns
        entry = pd.DataFrame(columns=columns)
        
        data_dict = buildFeatures(text)
        for name, variable in zip(["age","sign","gender","topic"],[age, sign, gender, topic]):
            if variable != None:
                if not name == target_variable:
                    data_dict[name] = variable
                else:
                    print("Error, target variable is not empty")
                
            elif name != target_variable:
                print("Error, feature missing")
        
        entry_numerical  = entry.append(data_dict, ignore_index=True)
        
        data_transformed =  self.pipeline_mapping[target_variable].\
                                transformation.numerical_transformer.transform(entry_numerical)
        
        return self.pipeline_mapping[target_variable].modelling.predict_numerical(data_transformed)[0]
            
        
    def predict_weighted(self, target_variable, text, age=None, sign=None, gender=None, topic=None):
        # Build dataframe for transformer
        data_dict = buildFeatures(text)
        data_dict["text_preprocessed"] = self.preprocessor.ProcessOne(text)
        for name, variable in zip(["age","sign","gender","topic"],[age, sign, gender, topic]):
            if variable != None:
                if not name == target_variable:
                    data_dict[name] = variable
                else:
                    print("Error, target variable is not empty")
                
            elif name != target_variable:
                print("Error, feature missing")
        
        prediction = self.pipeline_mapping[target_variable].modelling.weighted_prediction(data_dict,\
                                                                                          algo_type=self.pipeline_mapping[target_variable]\
                                                                                          .algo_type)
        return prediction[0]