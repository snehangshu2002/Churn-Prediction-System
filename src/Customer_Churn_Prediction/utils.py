import os
import sys
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
import pickle

def read_csv_data():
    logging.info("Reading CSV file started")
    try:
        logging.info("Import Successful")
        df = pd.read_csv(os.path.join("notebook", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex,sys)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with hyperparameter tuning for classification tasks
    """
    try:
        report = {}
        best_params_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            rs = RandomizedSearchCV(model, para, cv=5, scoring='accuracy', n_jobs=-1)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            best_params_report[list(models.keys())[i]] = rs.best_params_  # <-- save best params

        return report,best_params_report

    except Exception as e:
        raise CustomException(e, sys)