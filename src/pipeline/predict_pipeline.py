import os 
import sys
import pandas as pd
import numpy as np 
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self, age : int,  workclass, education_num : int, 
        marital_status, occupation,relationship,  race,
        sex, capital_gain:float,capital_loss:float,
        hours_per_week:float, native_country ):

        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex =sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def get_data_as_dataframe(self):
        try:
            custom_data_to_dict = {
                "age" : [self.age],
                "workclass" : [self.workclass],
                "education-num" : [self.education_num],
                "marital-status" : [self.marital_status],
                "occupation" : [self.occupation],
                "relationship" : [self.relationship],
                "race" : [self.race], 
                "sex" : [self.sex],
                "capital-gain" : [self.capital_gain], 
                "capital-loss" : [self.capital_loss], 
                "hours-per-week" : [self.hours_per_week],
                "native-country" : [self.native_country]
            }

            return pd.DataFrame(custom_data_to_dict)
        

        except Exception as e:
            raise CustomException(e,sys)