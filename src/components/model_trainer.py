import os 
import sys
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from src.exception import CustomException 
from src.logger import logging
from src.utils import model_evaluate,save_object

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

@dataclass 
class ModelTrainerConfig:
    model_obj_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models ={
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(solver='liblinear'),
                    "Kneighbors" : KNeighborsClassifier()
            }
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 10],
                    "criterion":['entropy','gini']
                },
                "Decision Tree": {
                    "criterion": ['gini', 'entropy'],
                    "splitter": ['best','random']
                },
                "Logistic Regression": {
                    "C": [5, 1, 10,20]
                },
                "Kneighbors": {
                    "n_neighbors": [3, 5, 10],
                    "weights": ['uniform', 'distance'],
                }
            }



            model_report: dict = model_evaluate(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score = max(list(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info("Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.model_obj_path,obj=best_model)

            return best_model_score,best_model_name



        except Exception as e:
            raise CustomException(e,sys)