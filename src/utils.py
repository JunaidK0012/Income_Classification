import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
import json
from sklearn.model_selection import GridSearchCV


def replace_special(input_df,columns):
    try:
        for x in columns:
            input_df[x].replace('?','Unknown',inplace=True)

        return input_df

    except Exception as e:
        raise CustomException(e,sys)
    

def remove_dot(x):
    try:
        if x[-1] == '.':
            return x[:-1]
        else:
            return x 
        
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object_json(obj,file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'w') as json_file:
            json.dump(obj,json_file)


    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(obj,file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

    
def get_columns(input_df,columns,target_column):
    try:
        num_columns=[]
        cat_columns=[]

        for x in columns:
            if x==target_column:
                pass
            else:
                if input_df[x].dtype == 'O':
                    cat_columns.append(x)
                else:
                    num_columns.append(x)

        return(num_columns,cat_columns)


    except Exception as e:
        raise CustomException(e,sys)
    
def model_evaluate(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            model_score = model.score(X_test,y_test)

            report[list(models.keys())[i]] = model_score

        return report


    except Exception as e:
        raise CustomException(e,sys)