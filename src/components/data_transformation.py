import os 
import sys
import pandas as pd
import numpy as np 

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,get_columns

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self,num_columns,cat_columns):
        try:
            cat_pipeline = Pipeline(
                steps=[
                    ('encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(transformers=[
                ('cat_columns',cat_pipeline,cat_columns),
                ('num_pipeline',num_pipeline,num_columns)
            ])

            return preprocessor
            


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data completed.")


            logging.info("Obtaining preprocessing object.")

            target_column = 'income'
            num_columns,cat_columns = get_columns(train_df,train_df.columns,target_column)


            preprocessor_obj = self.get_preprocessor_obj(num_columns,cat_columns) 

            input_feature_train_df = train_df.drop([target_column],axis='columns')   #X_train
            target_feature_train_df = train_df[target_column]                        #y_train

            input_feature_test_df = test_df.drop([target_column],axis='columns')     #X_test
            target_feature_test_df = test_df[target_column]                          #y_test

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path= self.data_transformation_config.preprocessor_obj_path,obj=preprocessor_obj)
            logging.info("Data Transformation Completed")
            
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        

        
        





