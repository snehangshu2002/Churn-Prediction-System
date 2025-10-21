import os
import sys
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
import pandas as pd


def read_csv_data():
    logging.info("Reading CSV file started")
    try:
        logging.info("Import Successful")
        df=pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)