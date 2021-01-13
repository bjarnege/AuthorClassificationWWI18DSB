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
    
    def __init__(self, pipelines, pipelines_cluster, pipeline_knn):
        # storing and unpacking the pipelines
        self.pipelines = pipelines
        self.pipeline_mapping = {"age": pipelines.values[0],
                                 "gender": pipelines.values[1],
                                 "sign": pipelines.values[2],
                                 "topic": pipelines.values[3]}
        
        # preprocessed dataset:
        self.df_full_preprocessed = pd.read_pickle("./df_full_preprocessed.pkl")

        # transformers for unsupervised algorithms (knn and KMeans)
        self.tfidf = pipelines_cluster[0]
        self.numerical_transformer = pipelines_cluster[1]
        
        # KMeans instances
        self.kmeans_text = pipelines_cluster[2]
        self.kmeans_numerical = pipelines_cluster[3]
        
        # KNearestNeighbors instances
        self.knn_text = pipeline_knn[2]
        self.knn_numerical = pipeline_knn[3]
        
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
    
    def transformation_preprocessor(self, mode, text=None, age=None, sign=None, gender=None, topic=None):
        if mode == "text":
            text_preprocessed = self.preprocessor.ProcessOne(text)
            return self.tfidf.transform([text_preprocessed])
        
        if mode == "numerical":
            features = buildFeatures(text)
            columns = {"gender": gender,
                        "age": age,
                        "topic": topic,
                        "sign": sign}

            data = {**columns, **features}
            data = pd.DataFrame(columns=data.keys()).append(data, ignore_index=True)
            return self.numerical_transformer.transform(data)
        
    def transform_cluster(self, mode="text", text=None, age=None, sign=None, gender=None, topic=None):
        if mode == "text":
            tfidf_text = self.transformation_preprocessor(mode, text)
            return self.kmeans_text.predict(tfidf_text)[0]

        if mode == "numerical":
            data_transformed = self.transformation_preprocessor(mode, text, age, sign, gender, topic)
            return self.kmeans_numerical.predict(data_transformed)[0]
        
    def transform_knn(self, mode="text", text=None, age=None, sign=None, gender=None, topic=None):
        if mode == "text":
            neighbors_indexes = self.knn_text.\
                                kneighbors(self.transformation_preprocessor(mode, text))
            return self.df_full_preprocessed.iloc[neighbors_indexes[1][0]]["text_preprocessed"]

        if mode == "numerical":
            neighbors_indexes = self.knn_numerical.\
                                kneighbors(self.transformation_preprocessor(mode, text, age, sign, gender, topic))
            return self.df_full_preprocessed.iloc[neighbors_indexes[1][0]]["text_preprocessed"]