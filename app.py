from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from src.Customer_Churn_Prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.Customer_Churn_Prediction.logger import logging
from src.Customer_Churn_Prediction.exception import CustomException
import uvicorn
import sys
import os

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API that predicts customer churn using machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates
templates = Jinja2Templates(directory="templates")

# Create Pydantic model for input data validation
class ChurnPredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.6,
                "TotalCharges": 1572.4
            }
        }

# Create root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return FileResponse("templates/index.html")

# Create prediction endpoint
@app.post("/predict")
async def predict_churn(data: ChurnPredictionInput):
    try:
        # Create CustomData instance
        custom_data = CustomData(
            gender=data.gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            tenure=data.tenure,
            PhoneService=data.PhoneService,
            MultipleLines=data.MultipleLines,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            OnlineBackup=data.OnlineBackup,
            DeviceProtection=data.DeviceProtection,
            TechSupport=data.TechSupport,
            StreamingTV=data.StreamingTV,
            StreamingMovies=data.StreamingMovies,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,
            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges
        )

        # Get prediction
        pred_df = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        # Return prediction
        churn_status = "Yes" if prediction[0] == 1 else "No"
        return {
            "churn_prediction": churn_status,
            "probability": int(prediction[0]),
            "message": f"The customer is {'likely' if prediction[0] == 1 else 'not likely'} to churn"
        }

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


