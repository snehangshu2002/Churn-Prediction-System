
# Customer Churn Prediction

This project provides a complete system to predict customer churn using a Telco dataset. It includes tools for data exploration, reusable code for data processing and model training, a prediction system, and deployment support using FastAPI and Docker.

For a detailed explanation of the project, including decisions and reproduction steps, see `PROJECT_DOCUMENTATION.md` in the repository root.

## Quick Links

- **Notebook (EDA & Experiments):** `customer.ipynb`
- **Python Code Package:** `src/Customer_Churn_Prediction`
- **Saved Models & Preprocessors:** `artifacts/`
- **Detailed Documentation:** `PROJECT_DOCUMENTATION.md`
- **Dockerfile:** `Dockerfile`

---

## Getting Started (Local Setup)

Follow these steps to run the project locally:

1. **Create a virtual environment** (to isolate dependencies):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the pipeline** (data loading, processing, training):
   ```powershell
   python src\Customer_Churn_Prediction\components\data_ingestion.py
   ```
   This generates:
   - `artifacts/raw.csv`: Raw dataset
   - `artifacts/train.csv`: Training data
   - `artifacts/test.csv`: Test data
   - `artifacts/preprocessor.pkl`: Preprocessing object
   - `artifacts/model.pkl`: Trained model

4. **Make a prediction** (example):
   ```powershell
   python - <<'PY'
   from src.Customer_Churn_Prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
   data = CustomData(
       gender='Male', SeniorCitizen=0, Partner='No', Dependents='No', tenure=12,
       PhoneService='Yes', MultipleLines='No', InternetService='Fiber optic',
       OnlineSecurity='No', OnlineBackup='No', DeviceProtection='No', TechSupport='No',
       StreamingTV='No', StreamingMovies='No', Contract='Month-to-month',
       PaperlessBilling='Yes', PaymentMethod='Electronic check', MonthlyCharges=70.35, TotalCharges=845.5
   )
   df = data.get_data_as_data_frame()
   pp = PredictPipeline()
   pred = pp.predict(df)
   print('Predicted churn (0=no, 1=yes):', pred)
   PY
   ```

---

## Running the Notebook

To explore the data and experiments:
- Open `customer.ipynb` in Jupyter Notebook or VS Code.
- Run the cells for exploratory data analysis (EDA), visualizations, and model experiments.

---

## Using Docker

Run the project as a Docker container for easy deployment.

1. **Build the Docker image**:
   ```powershell
   docker build -t snehangshu2002/churn-prediction-system:latest .
   ```

2. **Run the container** (on port 8000):
   ```powershell
   docker run --rm -p 8000:8000 snehangshu2002/churn-prediction-system:latest
   ```
   The container runs `app.py` by default. For FastAPI with Uvicorn, update `Dockerfile`:
   ```dockerfile
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Test the API**:
   - Check service status:
     ```powershell
     curl http://localhost:8000/health
     ```
   - Make a prediction (adjust JSON to match data):
     ```powershell
     curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"gender":"Male","SeniorCitizen":0,"tenure":12,...}'
     ```

---

## Project Structure

- **`customer.ipynb`**: Notebook for EDA and model experiments.
- **`src/Customer_Churn_Prediction/`**: Python package with:
  - `components/`: Scripts for data loading (`data_ingestion.py`), processing (`data_transformation.py`), and training (`model_trainer.py`).
  - `pipelines/`: Prediction logic (`prediction_pipeline.py` with `PredictPipeline`, `CustomData`).
  - `utils.py`, `logger.py`, `exception.py`: Helper functions.
- **`artifacts/`**: Stores CSVs and saved models/preprocessors.
- **`Dockerfile`**: Docker image build instructions.
- **`PROJECT_DOCUMENTATION.md`**: Detailed project guide.

---

## FastAPI Web Service

The project includes a FastAPI service for predictions with these features:

- **Endpoints**:
  - `GET /health`: Checks service status and model metadata.
  - `POST /predict`: Takes customer data (JSON) and returns churn predictions.

- **Details**:
  - Uses `PredictPipeline` to load `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.
  - Validates inputs with `CustomData` for correct formatting.
  - Docker image includes all files, but can be modified to fetch models from cloud storage (e.g., S3).

I can add `app.py` (FastAPI server) or update `Dockerfile` to use Uvicorn by default if needed.

---

## Contributing

Contributions are welcome! Suggested tasks:
- Add unit tests for `data_transformation.py` and `model_trainer.py`.
- Create `pipelines/training_pipeline.py` for command-line training.
- Build a production-ready `app.py` for FastAPI with S3 model loading.

Submit improvements via a pull request.

---

## License

This project is released under the MIT License. See the `LICENSE` file in the repository root for the full text.



