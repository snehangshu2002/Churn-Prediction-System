## Churn Prediction System — Project Documentation

This document describes the Churn Prediction System project, its structure, how to set it up, and step-by-step instructions to reproduce the data preparation, modeling pipelines, and notebook analyses included in this repository.

***

## Table of Contents

- Project overview
- Prerequisites
- Project structure (key files)
- Data description
- How the code is organized (modules & responsibilities)
- Step-by-step: reproduce the notebook EDA & modeling
- Step-by-step: run the packaged pipeline (ingestion → transformation → training → prediction)
- Deployment & serving (Hugging Face Spaces, FastAPI, Docker)
- Notes, assumptions, and next steps

***

## Project overview

This repository contains an end-to-end churn prediction system for a Telco customer dataset. It includes:

- Exploratory Data Analysis and model experiments in `customer.ipynb`.
- A modular Python package under `src/Customer_Churn_Prediction` implementing data ingestion, transformation, modeling, utilities, and simple pipelines.
- Artifacts (trained model, preprocessor) written to the `artifacts/` directory.
- A FastAPI-based serving layer for API deployment, containerized with Docker.
- Live deployment on Hugging Face Spaces for interactive predictions.

Primary goal: preprocess Telco customer data, handle imbalance with SMOTE, train and evaluate multiple classifiers (Logistic Regression, Random Forest, SVM, AdaBoost, XGBoost), persist the best model, and serve predictions via a web interface/API.[1][2]

***

## Prerequisites

- Python 3.8+ (recommend 3.9 or 3.10)
- The repository provides a `requirements.txt`. Install dependencies with:

```powershell
pip install -r requirements.txt
```

Note: If you use a virtual environment, activate it before installing. For Docker deployment, no local Python setup is required—build and run the container directly.

***

## Project structure (key files)

- `customer.ipynb` — Notebook with EDA, preprocessing, modeling experiments and model saving.
- `src/Customer_Churn_Prediction/` — Main package:
  - `exception.py` — Custom exception wrapper for consistent error messages.
  - `logger.py` — Logging configuration (writes logs into `logs/` with timestamped filename).
  - `utils.py` — Helpers: read CSV, save/load objects, and an evaluate_models helper used by training.
  - `components/data_ingestion.py` — Reads raw CSV (from `notebook/data`), writes `artifacts/raw.csv`, and splits to `artifacts/train.csv` and `artifacts/test.csv`.
  - `components/data_transformation.py` — Implements `TelcoDataCleaner` (custom transformer) and `DataTransformation` pipeline; handles type conversions, missing values, label encoding for some columns, scaling of numeric features, and SMOTE to balance the training set. Saves a combined preprocessor to `artifacts/preprocessor.pkl`.
  - `components/model_trainer.py` — Trains multiple classifiers, runs randomized search for hyperparameter tuning, picks the best model, logs metrics (mlflow/dagshub is initialized in the code), and saves the final model to `artifacts/model.pkl`.
  - `components/model_monitering.py` — (empty file placeholder in this repo)
  - `pipelines/prediction_pipeline.py` — `PredictPipeline` and `CustomData` helper to transform a single row and obtain predictions using saved `preprocessor.pkl` and `model.pkl` from `artifacts/`.
  - `pipelines/training_pipeline.py` — (empty file placeholder in this repo)

- `artifacts/` — CSVs and pickled objects produced by pipeline runs (train/test/raw, model.pkl, preprocessor.pkl).
- `app.py` — FastAPI application serving the model as a web API with endpoints for health checks and predictions.
- `templates/index.html` — Frontend HTML form for interactive churn predictions.
- `Dockerfile` — Containerizes the FastAPI app and model artifacts for deployment.
- `README.md` — Includes YAML metadata for Hugging Face Spaces (sdk: docker, app_port: 7860).

***

## Data

- The main dataset used for experiments is the Telco Customer Churn dataset located at `notebook/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Key columns:
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`
  - Several categorical columns (services, contract type, payment method, etc.) and `Churn` as target.

Notes from preprocessing:
- `customerID` is dropped.
- `TotalCharges` contains blanks and is coerced to numeric then filled with median.
- 'No internet service' and 'No phone service' categories are replaced with 'No'.
- Many Yes/No features are mapped to 1/0.
- `gender` mapped to 0/1, `SeniorCitizen` to integer.[2][1]

***

## Core module responsibilities (detailed)

1) `src/Customer_Churn_Prediction/utils.py`
   - read_csv_data(): reads the CSV used by notebook flows.
   - save_object(file_path, obj) / load_object(file_path): pickle persistence helpers.
   - evaluate_models(X_train, y_train, X_test, y_test, models, param): helper to run RandomizedSearchCV or Grid search per model and return scores and best parameters.

2) `src/Customer_Churn_Prediction/components/data_ingestion.py`
   - `DataIngestion.initiate_data_ingestion()`:
     - Reads CSV from `notebook/data`.
     - Writes `artifacts/raw.csv`.
     - Splits into train/test (80/20) and writes to `artifacts/train.csv` and `artifacts/test.csv`.
     - Returns paths to train and test CSV files.

3) `src/Customer_Churn_Prediction/components/data_transformation.py`
   - `TelcoDataCleaner`: custom scikit-learn compatible transformer that:
     - Drops `customerID`.
     - Converts `TotalCharges` to numeric and fills missing with median.
     - Replaces 'No internet service' / 'No phone service' with 'No'.
     - Maps Yes/No columns to 1/0 and `gender` to 0/1.
     - Label-encodes `InternetService`, `Contract`, `PaymentMethod` using fitted LabelEncoders.
   - `DataTransformation.get_data_transformer_object()`:
     - Builds a `ColumnTransformer` that scales numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) and passes through other features (cleaner handles most categoricals).
   - `initiate_data_transformation(train_path, test_path)`:
     - Reads train/test CSVs, fits `TelcoDataCleaner` on train features, transforms both datasets, encodes target (`Churn`) to 1/0, uses SMOTE to oversample the minority class on training features, applies scaling, and saves a `full_preprocessor` (a Pipeline combining cleaner + scaler) as `artifacts/preprocessor.pkl`.
     - Returns `train_arr`, `test_arr`, and the path to `preprocessor.pkl`.

4) `src/Customer_Churn_Prediction/components/model_trainer.py`
   - `ModelTrainer.initiate_model_trainer(train_array, test_array)`:
     - Splits arrays into X and y (last column is y).
     - Defines candidate models: Logistic Regression, Random Forest, SVM, AdaBoost, XGBoost.
     - Defines parameter grids and uses `evaluate_models` (RandomizedSearchCV) to tune and evaluate.
     - Logs metrics with MLflow/dagshub integration (dagshub.init present).
     - Saves the chosen best model to `artifacts/model.pkl` using `save_object`.

5) `src/Customer_Churn_Prediction/pipelines/prediction_pipeline.py`
   - `PredictPipeline.predict(features)` loads `artifacts/model.pkl` and `artifacts/preprocessor.pkl`, transforms input `features` and returns predictions.
   - `CustomData` is a convenience class to collect single-row inputs into a pandas DataFrame matching the model expected features.

***

## Step-by-step: reproduce the notebook EDA & modeling (recommended quick steps)

1) Install requirements (see Prerequisites above).

2) Open and run the notebook `customer.ipynb` to reproduce the exploratory steps. High-level steps in the notebook:

   - Load and inspect the dataset.
   - Clean categorical and numeric features.
   - Encode labels and apply SMOTE for imbalance.
   - Train baseline models and evaluate with cross-validation.
   - Save preprocessor and model artifacts.

3) For automated reproduction without the notebook, run the full pipeline:

   ```powershell
   # Run ingestion, transformation, and training
   python src/Customer_Churn_Prediction/components/data_ingestion.py
   python src/Customer_Churn_Prediction/components/data_transformation.py
   python src/Customer_Churn_Prediction/components/model_trainer.py
   ```

   This generates `artifacts/preprocessor.pkl` and `artifacts/model.pkl` for serving.

***

## Deployment & serving (Hugging Face Spaces, FastAPI, Docker)

The project is deployed as a Docker container on Hugging Face Spaces, serving predictions via FastAPI. The live demo is available at [https://huggingface.co/spaces/snehangshu270219/Customer_Churn_prediction_02](https://huggingface.co/spaces/snehangshu270219/Customer_Churn_prediction_02), where users can input customer data through an interactive web form and receive churn predictions.

### FastAPI Design
- **Endpoints**:
  - `GET /` — Serves the interactive HTML form (`templates/index.html`) for predictions.
  - `GET /health` — Returns server status and model readiness.
  - `POST /predict` — Accepts JSON payload matching `ChurnPredictionInput` model, uses `PredictPipeline` for inference, and returns churn prediction with probability and message.
- The app loads `artifacts/preprocessor.pkl` and `artifacts/model.pkl` at startup, validates inputs with Pydantic, and handles errors with custom exceptions.
- CORS is enabled for potential cross-origin calls, though the frontend runs same-origin in the Space.[3][1]

### Docker Setup
The `Dockerfile` builds a self-contained image with the FastAPI app, model artifacts, and dependencies. Key features:
- Runs as non-root user (uid 1000) for Spaces compatibility.
- Copies `artifacts/`, `templates/`, and source code with proper ownership.
- Installs from `requirements.txt` with pip caching.
- Exposes port 7860 and starts Uvicorn on `0.0.0.0:7860`.

To build and run locally:
1. Build: `docker build -t churn-prediction .`
2. Run: `docker run -p 7860:7860 --name churn-service churn-prediction`
3. Test: Visit `http://localhost:7860` for the form, or `curl -X POST http://localhost:7860/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,...}'`

Artifacts are baked into the image for simplicity; for production updates, store in persistent storage like S3 and download at startup.[4][2]

### Hugging Face Spaces Deployment
- **SDK Configuration**: README.md includes YAML front matter: `sdk: docker` and `app_port: 7860` for routing.
- **Build Process**: Pushing changes to the Space repo triggers automatic Docker builds. The container starts the FastAPI app, serving the web interface and API.
- **Interactive Demo**: The form at root (`/`) collects inputs, posts JSON to `/predict` (relative path for same-origin), and displays results without CORS issues.
- **Monitoring**: Use Spaces logs for debugging; the `/health` endpoint supports external health checks.
- **CI/CD**: For automation, integrate GitHub Actions to sync from a private repo to the Space (using HF_TOKEN secret and git push to the Space remote).[5][1]

Limitations in Spaces:
- Free tier has 2-hour idle timeouts; upgrade for persistent runtime.
- Artifacts are stored in the Space's disk (persistent across rebuilds) but limited to 50GB.
- Scale via hardware upgrades (CPU/GPU options) for higher traffic.[2][3]

### Alternative Deployments
For scaling beyond Spaces, deploy to AWS ECR + ECS/Fargate:
1. Push image to ECR: `docker tag churn-prediction:latest <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest && docker push ...`
2. Create ECS cluster/task/service with port 7860 mapping and IAM for S3 artifact access.
3. Use Application Load Balancer for traffic and CloudWatch for logs/metrics.[1][4]

***

## How I built the Telco Churn Prediction project — a step-by-step blog

This is a narrative of the work I completed in this repository: the exploratory analysis done in `customer.ipynb`, the core engineering decisions I made, the modeling experiments I ran, and how I packaged everything into a reproducible pipeline under `src/Customer_Churn_Prediction`.

If you prefer a quick summary, skip to the TL;DR. If you want to reproduce the work, follow the "How to reproduce" section and the code snippets below.

***

## TL;DR — what I accomplished

- Cleaned and preprocessed the Telco customer churn dataset.
- Consolidated categorical labels and handled numeric parsing issues.
- Used StandardScaler + SMOTE to prepare training data without leaking information from test to train.
- Trained and tuned multiple models (Logistic Regression, Random Forest, SVM, AdaBoost, XGBoost, Stacking and a small ANN). Evaluated them with ROC AUC and classification metrics.
- Packaged the pipeline (ingestion → transformation → training → prediction) into modular components and saved artifacts (`artifacts/preprocessor.pkl`, `artifacts/model.pkl`).
- Deployed as a Dockerized FastAPI app on Hugging Face Spaces for interactive and API-based predictions.

***

## Motivation and initial questions

I wanted a repeatable, auditable process that starts with a CSV and ends with a saved model and a prediction function. The notebook was my experimentation ground. Once experiments converged to reliable steps, I implemented them programmatically so they can be run again and integrated into other systems. Finally, I added a serving layer for real-world use.

Key questions I asked while working:
- How do I handle blanks in numeric fields like `TotalCharges`?
- Which categorical encoding strategy is robust while keeping inference simple?
- How do I fix class imbalance in a way that does not cause leakage?
- Which models provide the best trade-off between interpretability and performance?
- How to containerize and deploy for easy sharing and scaling?

***

## Data cleaning & preprocessing — what I actually did

1) Drop non-informative ID column

I removed `customerID` because it's an identifier not predictive of churn.

2) Fix `TotalCharges` and blanks

`TotalCharges` sometimes contains an empty string; I converted the column to numeric (coercing errors to NaN) and imputed missing values with the median.

Code snippet:

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

3) Simplify redundant categories

Columns like `OnlineBackup` or `MultipleLines` used a special token `No internet service` or `No phone service`. I replaced these with `No` to reduce unique categories and make encoding simpler.

```python
df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
```

4) Encode Yes/No and binary fields

All Yes/No fields (including `Churn`) were mapped to 1/0. `gender` was mapped to 0/1 as well.

5) Label-encode remaining categorical features

I fitted `LabelEncoder` for object columns such as `InternetService`, `Contract`, and `PaymentMethod`. In the production pipeline a consistent encoder is fit on the training data and used at inference time.

6) Scale numeric features

I used `StandardScaler` on `tenure`, `MonthlyCharges`, and `TotalCharges`.

7) Handle class imbalance safely with SMOTE

SMOTE was applied only to the training features after scaling. This balances classes for model fitting while keeping the test set as a realistic holdout.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```

Important: Do not fit SMOTE on the full dataset before split — that leaks synthetic samples into the test set.

***

## Modeling experiments — what I tried and why

I experimented with a variety of models to find a balance between performance and interpretability.

- Logistic Regression — baseline, interpretable, quick to tune.
- Random Forest — robust, handles mixed feature types and non-linearity.
- XGBoost — typically strong performance with tabular data.
- AdaBoost — ensemble of weak learners to boost recall/precision.
- SVM — alternative non-linear option (slower with large feature sets).
- Stacking — combined multiple strong learners into a stacked classifier.
- ANN — small feed-forward network as a non-tree baseline.

Hyperparameter search: I used GridSearchCV / RandomizedSearchCV with StratifiedKFold to avoid class imbalance issues in CV splits.

Evaluation: I tracked ROC AUC, accuracy, precision, recall, F1 and confusion matrices. I also experimented with decision thresholds on probabilities to prioritize precision or recall.

Code excerpt (example evaluate / refit loop):

```python
# utils.evaluate_models is used to run randomized/grid search per model
models = { 'Logistic Regression': LogisticRegression(...), 'Random Forest': RandomForestClassifier(...), ... }
params = { 'Logistic Regression': {...}, 'Random Forest': {...}, ... }
report, best_params = evaluate_models(X_train, y_train, X_test, y_test, models, params)
```

The training component selects the best model by test score and saves it with `save_object(..., 'artifacts/model.pkl')`.

***

## Engineering the pipeline — files and flow

I converted the notebook steps into the following components:

- `src/Customer_Churn_Prediction/components/data_ingestion.py` — reads the raw CSV and creates `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/raw.csv`.
- `src/Customer_Churn_Prediction/components/data_transformation.py` — `TelcoDataCleaner` implements cleaning rules and is combined with a numeric pipeline (impute + scaler). The combined preprocessor is saved as `artifacts/preprocessor.pkl`.
- `src/Customer_Churn_Prediction/components/model_trainer.py` — orchestrates model training, hyperparameter search, mlflow logging (dagshub integration present) and persists the best model to `artifacts/model.pkl`.
- `src/Customer_Churn_Prediction/pipelines/prediction_pipeline.py` — `PredictPipeline` loads preprocessor + model and predicts on single-row inputs; `CustomData` helps create that input.

Why this module layout?

It separates concerns:
- ingestion handles file IO,
- transformation handles feature engineering and ensures the exact same steps are available at inference,
- trainer focuses on model selection and persistence,
- prediction pipeline is a light wrapper for inference.

***

## How to reproduce everything locally (copy-paste commands)

From the repository root (Windows PowerShell):

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full example (ingestion → transformation → training):

```powershell
python src/Customer_Churn_Prediction/components/data_ingestion.py
```

This script will create `artifacts/train.csv`, `artifacts/test.csv`, save `artifacts/preprocessor.pkl`, and save `artifacts/model.pkl` after training.

Run only data ingestion (if you want to inspect CSV artifacts first):

```powershell
python -c "from src.Customer_Churn_Prediction.components.data_ingestion import DataIngestion; DataIngestion().initiate_data_ingestion()"
```

Run only transformation (after ingestion):

```powershell
python -c "from src.Customer_Churn_Prediction.components.data_transformation import DataTransformation; DataTransformation().initiate_data_transformation('artifacts/train.csv', 'artifacts/test.csv')"
```

Make a single prediction (requires `artifacts/preprocessor.pkl` and `artifacts/model.pkl`):

```powershell
python - <<'PY'
from src.Customer_Churn_Prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
data = CustomData(gender='Male', SeniorCitizen=0, Partner='No', Dependents='No', tenure=12,
                  PhoneService='Yes', MultipleLines='No', InternetService='Fiber optic',
                  OnlineSecurity='No', OnlineBackup='No', DeviceProtection='No', TechSupport='No',
                  StreamingTV='No', StreamingMovies='No', Contract='Month-to-month',
                  PaperlessBilling='Yes', PaymentMethod='Electronic check', MonthlyCharges=70.35, TotalCharges=845.5)
df = data.get_data_as_data_frame()
pp = PredictPipeline()
pred = pp.predict(df)
print('Predicted churn (0=no,1=yes):', pred)
PY
```

Test the FastAPI app locally (after running the pipeline):

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` for the form or `http://localhost:8000/docs` for API testing.

For Docker: `docker build -t churn-prediction . && docker run -p 7860:7860 churn-prediction`

***

## Practical lessons and decisions I defend

- Keep cleaning logic in a transformer class so training and inference are consistent.
- Always apply SMOTE only to the training set after scaling.
- Save both the preprocessor and the model so they can be loaded reliably at inference.
- Use cross-validated randomized search when exploring many hyperparameters — it's faster than a full grid and usually finds good settings.
- Deploy with Docker and relative paths for frontend-backend communication to avoid CORS in cloud environments like Hugging Face Spaces.[1][2]

***

## Results & variability

During experiments, tree-based models (Random Forest, XGBoost), stacking and well-tuned ensembles delivered the strongest performance in terms of AUC and precision/recall trade-offs. Exact metrics vary per run — to reproduce exact numbers pin random seeds and persist the final fitted estimator.

The deployed Space handles real-time predictions with low latency, suitable for interactive demos and basic API use.

***

## Recommended next steps (concrete)

1) Add a small CLI (e.g., `run_pipeline.py`) that orchestrates the pipeline with flags (data path, test_size, seed, model selection). This makes the project easier for other developers.

2) Add unit tests for each component. Start with small fixtures and synthetic CSVs so CI runs fast.

3) Implement a minimal monitoring script in `model_monitering.py` to compute basic production metrics and data drift checks.

4) Save a JSON schema for input features and validate `CustomData` inputs before prediction.

5) Set up GitHub Actions for CI/CD: automate artifact building and sync to Hugging Face Spaces on push to main.

If you'd like, I can implement any of these for you — which would you prefer next?

