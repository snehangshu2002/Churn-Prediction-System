import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.Customer_Churn_Prediction.utils import save_object,evaluate_models
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred, pred_proba=None):
        """
        Calculate classification metrics
        """
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        
        # ROC AUC (if probabilities are available)
        if pred_proba is not None:
            roc_auc = roc_auc_score(actual, pred_proba)
        else:
            roc_auc = roc_auc_score(actual, pred)
            
        return accuracy, precision, recall, f1, roc_auc

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(random_state=42, probability=True),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss')
            }
            
            params = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 200, 500]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "SVM": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "XGBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }

            model_report, best_params_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            best_params = best_params_report[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model

            best_params = best_params_report[actual_model]

            # MLflow tracking (update with your own MLflow server URL)
            mlflow.set_registry_uri("https://dagshub.com/snehangshu2002/Churn-Prediction-System.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                
                # Get probabilities for ROC AUC if available
                if hasattr(best_model, "predict_proba"):
                    predicted_proba = best_model.predict_proba(X_test)[:, 1]
                else:
                    predicted_proba = None

                (accuracy, precision, recall, f1, roc_auc) = self.eval_metrics(
                    y_test, predicted_qualities, predicted_proba
                )

                mlflow.log_params(best_params)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.7:  # Adjusted threshold for classification
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy_score_final = accuracy_score(y_test, predicted)
            
            return accuracy_score_final

        except Exception as e:
            raise CustomException(e, sys)
