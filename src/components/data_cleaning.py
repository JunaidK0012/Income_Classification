import os
import sys 
import numpy as np 
import pandas as pd 
import json

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException 
from src.logger import logging
from src.utils import replace_special,remove_dot,save_object_json


@dataclass
class DataCleaningConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    clean_data_path = os.path.join('artifacts','data.csv')
    education_dict_path = os.path.join('artifacts','education.json')

class DataCleaning:
    def __init__(self):
        self.data_cleaning_config = DataCleaningConfig()

    def initiate_data_cleaning(self,raw_data_path):
        logging.info("Entered the data cleaning process")

        try:
            df = pd.read_csv(raw_data_path)
            logging.info("Filling Na values")
            df.fillna('Missing',inplace=True)

            logging.info("Removing duplicate values")
            df.drop_duplicates(inplace=True)


            logging.info("Replacing ? with Unknown")
            columns = ['workclass','occupation','native-country']
            df = replace_special(df,columns=columns)


            logging.info("Removing dot from income")
            df['income'] = df['income'].apply(remove_dot)
            


            education_dict = dict(zip(df['education'], df['education-num']))
            save_object_json(obj= education_dict,file_path=self.data_cleaning_config.education_dict_path)


            logging.info('Dropping unwanted columns')
            df.drop(['education','fnlwgt'],axis='columns',inplace=True)


            df.to_csv(self.data_cleaning_config.clean_data_path,index=False,header=True)

            logging.info("Initiating train test split")
            train_df,test_df = train_test_split(df,train_size=0.8,random_state=45)

            train_df.to_csv(self.data_cleaning_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.data_cleaning_config.test_data_path,index=False,header=True)
            logging.info("train-test split completed")


            return(self.data_cleaning_config.train_data_path,self.data_cleaning_config.test_data_path)


        except Exception as e:
            raise CustomException(e,sys)