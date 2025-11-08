from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
from datetime import datetime
import logging
import os
import json
import numpy as np
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
encoder = None
scaler = None
model_metrics = None


# Pydantic Models
class PropertyData(BaseModel):
    """Property data model for prediction"""
    municipality: str = Field(..., description="Municipality (e.g., Центар, Кисела Вода)")
    area: float = Field(..., gt=0, description="Property area in square meters")
    number_of_rooms: int = Field(..., ge=1, description="Number of rooms")

    # Binary features (0 or 1) - English property names with Cyrillic aliases
    balcony: int = Field(0, alias="Балкон / Тераса", ge=0, le=1)
    elevator: int = Field(0, alias="Лифт", ge=0, le=1)
    ground_floor: int = Field(0, alias="Приземје", ge=0, le=1)
    parking: int = Field(0, alias="Паркинг простор / Гаража", ge=0, le=1)
    loft: int = Field(0, alias="Поткровје", ge=0, le=1)
    new_building: int = Field(0, alias="Нова градба", ge=0, le=1)
    renovated: int = Field(0, alias="Реновиран", ge=0, le=1)
    furnished: int = Field(0, alias="Наместен", ge=0, le=1)
    basement: int = Field(0, alias="Подрум", ge=0, le=1)
    interphone: int = Field(0, alias="Интерфон", ge=0, le=1)
    duplex: int = Field(0, alias="Дуплекс", ge=0, le=1)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "municipality": "Центар",
                "area": 67,
                "number_of_rooms": 2,
                "Балкон / Тераса": 0,
                "Лифт": 1,
                "Приземје": 0,
                "Паркинг простор / Гаража": 0,
                "Поткровје": 0,
                "Нова градба": 0,
                "Реновиран": 1,
                "Наместен": 1,
                "Подрум": 1,
                "Интерфон": 1,
                "Дуплекс": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response model with variable confidence metrics"""
    predicted_price: float
    price_per_square_meter: float
    currency: str = "EUR"
    success: bool = True
    prediction_confidence: Dict[str, Any]
    model_accuracy: Dict[str, Any]
    model_info: Dict[str, Any]
    property_summary: Optional[Dict[str, Any]] = None


class BatchPropertyData(BaseModel):
    properties: List[PropertyData] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    success: bool
    total_properties: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    timestamp: str
    mean_absolute_error_eur: Optional[float] = None
    mae_percentage: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_estimators: Optional[int]
    max_depth: Optional[int]
    n_features: Optional[int]
    supported_municipalities: List[str]
    feature_categories: Dict[str, List[str]]
    mean_absolute_error_eur: Optional[float] = None
    mae_percentage: Optional[float] = None
    trained_on: Optional[str] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None
    service_info: Dict[str, str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI ML Service...")
    load_models()
    yield
    logger.info("Shutting down FastAPI ML Service...")


app = FastAPI(
    title="Property Price Prediction API",
    description="FastAPI ML service for predicting property prices in Skopje, North Macedonia",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models():
    global model, encoder, scaler, model_metrics

    try:
        model_dir = os.getenv("MODEL_DIR", "./models")

        model_path = os.path.join(model_dir, "price_prediction_m1.pkl")
        encoder_path = os.path.join(model_dir, "encoder.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        metrics_path = os.path.join(model_dir, "model_metrics.pkl")

        logger.info(f"Loading models from: {model_dir}")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)

        if os.path.exists(metrics_path):
            model_metrics = joblib.load(metrics_path)
            logger.info("✅ Model metrics loaded successfully")
            logger.info(
                f"   MAE: {model_metrics.get('mean_absolute_error_eur')} EUR ({model_metrics.get('mae_percentage')}%)")
        else:
            logger.warning("⚠️  Model metrics file not found")
            model_metrics = None

        logger.info("✅ Models loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Features: {model.n_features_in_}")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model = encoder = scaler = model_metrics = None


def preprocess_property_data(property_data: PropertyData) -> pd.DataFrame:
    """Preprocess property data for prediction"""
    data_dict = {
        'municipality': property_data.municipality,
        'area': property_data.area,
        'number_of_rooms': property_data.number_of_rooms,
        'Балкон / Тераса': property_data.balcony,
        'Лифт': property_data.elevator,
        'Приземје': property_data.ground_floor,
        'Паркинг простор / Гаража': property_data.parking,
        'Поткровје': property_data.loft,
        'Нова градба': property_data.new_building,
        'Реновиран': property_data.renovated,
        'Наместен': property_data.furnished,
        'Подрум': property_data.basement,
        'Интерфон': property_data.interphone,
        'Дуплекс': property_data.duplex
    }

    df = pd.DataFrame([data_dict])

    current_date = datetime.now()
    df['year'] = current_date.year
    df['month'] = current_date.month
    df['weekday'] = current_date.weekday()

    categorical_cols = ['municipality']
    binary_cols = ['Балкон / Тераса', 'Лифт', 'Приземје', 'Паркинг простор / Гаража',
                   'Поткровје', 'Нова градба', 'Реновиран', 'Наместен', 'Подрум',
                   'Интерфон', 'Дуплекс']
    numerical_cols = ['area', 'number_of_rooms', 'year', 'month', 'weekday']

    df_cat = pd.DataFrame(encoder.transform(df[categorical_cols]))
    df_cat.columns = encoder.get_feature_names_out(categorical_cols)

    df_num = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols)
    df_binary = df[binary_cols].reset_index(drop=True)

    processed_data = pd.concat([df_num, df_cat, df_binary], axis=1)
    return processed_data


def calculate_prediction_confidence(model, processed_data, property_data):
    """
    Calculate variable confidence metrics based on Random Forest tree predictions
    This CHANGES for each property based on how consistent the trees are
    """
    # Get predictions from all individual trees in the Random Forest
    tree_predictions = np.array([tree.predict(processed_data)[0]
                                 for tree in model.estimators_])

    # Calculate statistics
    mean_pred = tree_predictions.mean()
    std_pred = tree_predictions.std()
    min_pred = tree_predictions.min()
    max_pred = tree_predictions.max()

    # Calculate confidence score (0-100%)
    # Lower std = higher confidence
    # Typical std for property prices might be 5000-50000 EUR
    coefficient_of_variation = (std_pred / mean_pred) * 100 if mean_pred > 0 else 100

    # Confidence score: 100% = very confident, 0% = not confident
    confidence_score = max(0, 100 - coefficient_of_variation)

    # Confidence level category
    if confidence_score >= 80:
        confidence_level = "High"
    elif confidence_score >= 60:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    # Prediction range (using standard deviation)
    prediction_range_low = mean_pred - std_pred
    prediction_range_high = mean_pred + std_pred

    return {
        "confidence_score": round(confidence_score, 2),
        "confidence_level": confidence_level,
        "prediction_std_deviation": round(std_pred, 2),
        "prediction_range": {
            "low": round(prediction_range_low, 2),
            "high": round(prediction_range_high, 2),
            "description": "68% of tree predictions fall within this range"
        },
        "tree_predictions_range": {
            "min": round(min_pred, 2),
            "max": round(max_pred, 2)
        },
        "explanation": f"Based on {len(tree_predictions)} decision trees. {confidence_level} confidence means predictions are {'very consistent' if confidence_score >= 80 else 'somewhat varied' if confidence_score >= 60 else 'highly varied'} across trees."
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Property Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if all([model, encoder, scaler]) else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=all([model, encoder, scaler]),
        service="FastAPI ML Prediction Service",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(property_data: PropertyData):
    """
    Predict property price with variable confidence metrics

    Returns:
    - predicted_price: The predicted price
    - prediction_confidence: Variable confidence metrics (changes per request)
    - model_accuracy: Fixed accuracy metrics from training
    """

    if not all([model, encoder, scaler]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded properly"
        )

    try:
        # Preprocess data
        processed_data = preprocess_property_data(property_data)

        # Make prediction
        predicted_price = model.predict(processed_data)[0]
        price_per_square = predicted_price / property_data.area

        # Calculate variable confidence (CHANGES per request)
        prediction_confidence = calculate_prediction_confidence(
            model, processed_data, property_data
        )

        logger.info(
            f"Prediction: {predicted_price:.2f} EUR for {property_data.area}m² "
            f"in {property_data.municipality} (Confidence: {prediction_confidence['confidence_score']}%)"
        )

        # Fixed model accuracy (from training)
        model_accuracy = {
            "mean_absolute_error_eur": model_metrics.get('mean_absolute_error_eur') if model_metrics else None,
            "mae_percentage": model_metrics.get('mae_percentage') if model_metrics else None,
            "description": "Average prediction error on test data during training"
        }

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            price_per_square_meter=round(price_per_square, 2),
            currency="EUR",
            success=True,
            prediction_confidence=prediction_confidence,  # ← VARIABLE per request
            model_accuracy=model_accuracy,  # ← FIXED from training
            model_info={
                "model_type": type(model).__name__,
                "features_used": len(processed_data.columns),
                "prediction_timestamp": datetime.now().isoformat(),
                "service": "FastAPI"
            },
            property_summary={
                "municipality": property_data.municipality,
                "area": property_data.area,
                "rooms": property_data.number_of_rooms
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_data: BatchPropertyData):
    if not all([model, encoder, scaler]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded properly"
        )

    results = []
    errors = []

    for i, property_data in enumerate(batch_data.properties):
        try:
            processed_data = preprocess_property_data(property_data)
            predicted_price = model.predict(processed_data)[0]
            price_per_square = predicted_price / property_data.area

            # Add confidence for each property
            confidence = calculate_prediction_confidence(model, processed_data, property_data)

            results.append({
                "index": i,
                "predicted_price": round(predicted_price, 2),
                "price_per_square_meter": round(price_per_square, 2),
                "currency": "EUR",
                "municipality": property_data.municipality,
                "area": property_data.area,
                "confidence_score": confidence["confidence_score"],
                "confidence_level": confidence["confidence_level"]
            })

        except Exception as e:
            logger.error(f"Batch prediction error at index {i}: {str(e)}")
            errors.append({
                "index": i,
                "error": f"Prediction failed: {str(e)}"
            })

    logger.info(
        f"Batch prediction: {len(results)} successful, {len(errors)} failed "
        f"out of {len(batch_data.properties)}"
    )

    return BatchPredictionResponse(
        success=True,
        total_properties=len(batch_data.properties),
        successful_predictions=len(results),
        failed_predictions=len(errors),
        results=results,
        errors=errors,
        timestamp=datetime.now().isoformat(),
        mean_absolute_error_eur=model_metrics.get('mean_absolute_error_eur') if model_metrics else None,
        mae_percentage=model_metrics.get('mae_percentage') if model_metrics else None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    if not all([model, encoder, scaler]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    try:
        supported_municipalities = encoder.categories_[0].tolist()

        return ModelInfoResponse(
            model_type=type(model).__name__,
            n_estimators=getattr(model, 'n_estimators', None),
            max_depth=getattr(model, 'max_depth', None),
            n_features=getattr(model, 'n_features_in_', None),
            supported_municipalities=supported_municipalities,
            mean_absolute_error_eur=model_metrics.get('mean_absolute_error_eur') if model_metrics else None,
            mae_percentage=model_metrics.get('mae_percentage') if model_metrics else None,
            trained_on=None,
            training_samples=None,
            test_samples=None,
            feature_categories={
                "categorical": ["municipality"],
                "numerical": ["area", "number_of_rooms", "year", "month", "weekday"],
                "binary": [
                    "Балкон / Тераса", "Лифт", "Приземје", "Паркинг простор / Гаража",
                    "Поткровје", "Нова градба", "Реновиран", "Наместен",
                    "Подрум", "Интерфон", "Дуплекс"
                ]
            },
            service_info={
                "framework": "FastAPI",
                "version": "1.0.0",
                "python_version": os.sys.version.split()[0]
            }
        )

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@app.get("/municipalities", tags=["Model"])
async def get_municipalities():
    if not encoder:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder not loaded"
        )

    try:
        municipalities = encoder.categories_[0].tolist()

        return {
            "municipalities": municipalities,
            "total_count": len(municipalities),
            "note": "These are the municipalities the model was trained on"
        }

    except Exception as e:
        logger.error(f"Error getting municipalities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get municipalities: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )