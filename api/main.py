"""
Step 10: FastAPI Prediction API
================================
Production-ready REST API for churn prediction
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import io
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import predict_customer, predict_batch

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
## 🔮 Customer Churn Prediction Service

Powered by a Deep ANN trained on Telco Customer data.

### Endpoints:
- **POST /predict** — Predict churn for a single customer
- **POST /predict/batch** — Batch prediction via CSV upload
- **GET /health** — Health check
- **GET /model-info** — Model metadata

### Risk Levels:
- 🔴 **HIGH** (≥70%) — Immediate intervention required
- 🟡 **MEDIUM** (40–70%) — Engagement campaign
- 🟢 **LOW** (<40%) — Maintain relationship
    """,
    version="1.0.0",
    contact={"name": "ML Team", "email": "ml@company.com"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class CustomerInput(BaseModel):
    """Customer features for churn prediction."""
    gender: str = Field(..., example="Female", description="Male/Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0, description="0=No, 1=Yes")
    Partner: str = Field(..., example="Yes", description="Yes/No")
    Dependents: str = Field(..., example="No", description="Yes/No")
    tenure: int = Field(..., ge=0, example=12, description="Months with company")
    PhoneService: str = Field(..., example="Yes", description="Yes/No")
    MultipleLines: str = Field(..., example="No", description="Yes/No/No phone service")
    InternetService: str = Field(..., example="Fiber optic",
                                  description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month",
                           description="Month-to-month/One year/Two year")
    PaperlessBilling: str = Field(..., example="Yes", description="Yes/No")
    PaymentMethod: str = Field(..., example="Electronic check",
                                description="Electronic check/Mailed check/Bank transfer/Credit card")
    MonthlyCharges: float = Field(..., gt=0, example=79.85)
    TotalCharges: float = Field(..., ge=0, example=958.2)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 79.85, "TotalCharges": 958.2
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_percentage: str
    risk_level: str
    risk_color: str
    explanations: List[str]
    recommended_action: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Churn Prediction API", "version": "1.0.0"}


@app.get("/model-info", tags=["System"])
def model_info():
    """Model metadata and performance info."""
    return {
        "model_type": "Artificial Neural Network (ANN)",
        "framework": "TensorFlow/Keras",
        "architecture": "4 hidden layers (256-128-64-32)",
        "dataset": "IBM Telco Customer Churn",
        "features": 19,
        "target": "Churn (binary)",
        "metrics": {
            "accuracy": "~82%",
            "roc_auc": "~86%",
            "recall": "~80%"
        },
        "threshold": 0.5,
        "explainability": "SHAP DeepExplainer"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: 0–1 score
    - **risk_level**: HIGH / MEDIUM / LOW
    - **explanations**: Top factors driving churn risk
    - **recommended_action**: Business retention suggestion
    """
    try:
        result = predict_customer(customer.dict())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Batch prediction from CSV file upload.
    CSV must contain the same columns as the single prediction input.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        results = predict_batch(df)

        # Add to original df
        df['churn_probability'] = results['churn_probability']
        df['risk_level'] = results['risk_level']
        df['recommended_action'] = results['recommended_action']

        return {
            "total_customers": len(df),
            "high_risk": int((results['risk_level'] == 'HIGH').sum()),
            "medium_risk": int((results['risk_level'] == 'MEDIUM').sum()),
            "low_risk": int((results['risk_level'] == 'LOW').sum()),
            "predictions": results.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/what-if", tags=["Analysis"])
def what_if_analysis(customer: CustomerInput, monthly_charge_reduction: float = 10.0):
    """
    What-If Analysis: What happens to churn probability if we reduce monthly charges?
    Example: 10% reduction in MonthlyCharges → new churn probability
    """
    try:
        original = predict_customer(customer.dict())

        # Modified customer
        modified = customer.dict()
        reduction_amount = customer.MonthlyCharges * (monthly_charge_reduction / 100)
        modified['MonthlyCharges'] = customer.MonthlyCharges - reduction_amount
        modified_result = predict_customer(modified)

        delta = original['churn_probability'] - modified_result['churn_probability']

        return {
            "original_probability": original['churn_probability'],
            "original_risk": original['risk_level'],
            "modified_probability": modified_result['churn_probability'],
            "modified_risk": modified_result['risk_level'],
            "reduction_applied": f"{monthly_charge_reduction}% on MonthlyCharges",
            "probability_change": round(delta, 4),
            "impact": f"Churn probability {'decreased' if delta > 0 else 'increased'} by {abs(delta)*100:.1f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
