import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

from src.Customer_Churn_Prediction.utils import save_object
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class TelcoDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Telco customer data cleaning operations
    """
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        # Fit label encoders for categorical columns that need encoding
        cat_cols_to_encode = ["InternetService", "Contract", "PaymentMethod"]
        for col in cat_cols_to_encode:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col])
                self.label_encoders[col] = le
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Drop customerID if it exists (not needed for modeling)
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)
        
        # Handle blank TotalCharges
        if 'TotalCharges' in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors='coerce')
            X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)
        
        # Replace redundant service categories
        X.replace({
            'No internet service': 'No',
            'No phone service': 'No'
        }, inplace=True)
        
        # Convert Yes/No columns to binary (1/0)
        yes_no_columns = [
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
        ]
        
        # Only convert columns that exist in the dataframe
        existing_yes_no_cols = [col for col in yes_no_columns if col in X.columns]
        X[existing_yes_no_cols] = X[existing_yes_no_cols].replace({'Yes': 1, 'No': 0})
        
        # Convert gender to binary
        if 'gender' in X.columns:
            X["gender"] = X["gender"].map({'Female': 0, 'Male': 1})
        
        # Convert SeniorCitizen to integer if it's not already
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = X['SeniorCitizen'].astype(int)
        
        # Apply label encoding to remaining categorical columns
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])
        
        return X


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing pipeline
        """
        try:
            
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
            
            # After cleaning, these columns will be numerical
            # InternetService, Contract, PaymentMethod will be label encoded
            # All Yes/No columns will be converted to 1/0
            categorical_columns = []  # No categorical columns left after cleaning
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            logging.info(f"Numerical columns for scaling: {numerical_columns}")
            logging.info("All categorical data will be handled by TelcoDataCleaner")

            # Since all categorical data is handled by the cleaner, 
            # we only need numerical pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ],
                remainder='passthrough'  # Pass through all other columns as-is
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Main method that orchestrates the entire data transformation process
        """
        try:
            # Read the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test data completed")

            # Create and fit the data cleaner
            data_cleaner = TelcoDataCleaner()
            
            # Separate features and target before cleaning
            target_column_name = "Churn"
            
            # Clean the data (excluding target column for fitting)
            train_features = train_df.drop(columns=[target_column_name])
            test_features = test_df.drop(columns=[target_column_name])
            
            # Fit cleaner on training features only
            data_cleaner.fit(train_features)
            
            # Transform both train and test features
            train_features_cleaned = data_cleaner.transform(train_features)
            test_features_cleaned = data_cleaner.transform(test_features)
            
            # Handle target column - convert to binary if needed
            train_target = train_df[target_column_name].replace({'Yes': 1, 'No': 0})
            test_target = test_df[target_column_name].replace({'Yes': 1, 'No': 0})

            logging.info("Data cleaning completed")

            # Get preprocessing object for scaling
            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing (scaling)
            logging.info("Applying preprocessing (scaling) on cleaned data")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(train_features_cleaned)
            input_feature_test_arr = preprocessing_obj.transform(test_features_cleaned)

            # Apply SMOTE for handling class imbalance
            logging.info("Applying SMOTE for class balance")
            smote = SMOTE(random_state=42)
            x_train_resampled, y_train_resampled = smote.fit_resample(
                input_feature_train_arr, train_target
            )

            # Combine features and target
            train_arr = np.c_[x_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(test_target)]

            logging.info("Saving preprocessing objects")

            # Create a combined preprocessor that includes both cleaning and scaling
            from sklearn.pipeline import Pipeline
            full_preprocessor = Pipeline([
                ('cleaner', data_cleaner),
                ('scaler', preprocessing_obj)
            ])

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=full_preprocessor
            )

            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # For testing purposes
    pass
