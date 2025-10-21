from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
import sys
from src.Customer_Churn_Prediction.components.data_ingestion import DataIngestion,DataIngestionConfig
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)