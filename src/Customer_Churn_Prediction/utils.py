import os
import sys
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
import pandas as pd
import pickle

def read_csv_data():
    logging.info("Reading CSV file started")
    try:
        logging.info("Import Successful")
        df=pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)