from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import logging
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
model = None
encoder = None
scaler = None


# Pydantic Models
class PropertyData(BaseModel):
    """Property data model for prediction"""
    municipality: str = Field(..., description="Municipality (e.g., Центар, Кисела Вода)")
    area: float = Field(..., gt=0, description="Property area in square meters")
    number_of_rooms: int = Field(..., ge=1, description="Number of rooms")

    # Binary features (0 or 1) - using aliases for Cyrillic names
    balkon_terasa: int = Field(0, alias="Балкон / Тераса", ge=0, le=1)
    lift: int = Field(0, alias="Лифт", ge=0, le=1)
    prizemje: int = Field(0, alias="Приземје", ge=0, le=1)
    parking: int = Field(0, alias="Паркинг простор / Гаража", ge=0, le=1)
    potkrovje: int = Field(0, alias="Поткровје", ge=0, le=1)
    nova_gradba: int = Field(0, alias="Нова градба", ge=0, le=1)
    renoviran: int = Field(0, alias="Реновиран", ge=0, le=1)
    namesten: int = Field(0, alias="Наместен", ge=0, le=1)
    podrum: int = Field(0, alias="Подрум", ge=0, le=1)
    interfon: int = Field(0, alias="Интерфон", ge=0, le=1)
    dupleks: int = Field(0, alias="Дуплекс", ge=0, le=1)

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


class BatchPropertyData(BaseModel):
    """Batch prediction request model"""
    properties: List[PropertyData] = Field(..., max_length=100)


class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price: float
    price_per_square_meter: float
    currency: str = "EUR"
    success: bool = True
    model_info: Dict[str, Any]
    property_summary: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool
    total_properties: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    model_loaded: bool
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    n_estimators: Optional[int]
    max_depth: Optional[int]
    n_features: Optional[int]
    supported_municipalities: List[str]
    feature_categories: Dict[str, List[str]]
    service_info: Dict[str, str]


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    # Startup
    logger.info("Starting FastAPI ML Service...")
    load_models()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI ML Service...")


# Initialize FastAPI app
app = FastAPI(
    title="Property Price Prediction API",
    description="FastAPI ML service for predicting property prices in Skopje, North Macedonia",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models():
    """Load trained models and preprocessors"""
    global model, encoder, scaler

    try:
        model_dir = os.getenv("MODEL_DIR", "./models")

        model_path = os.path.join(model_dir, "price_prediction_rf_model.pkl")
        encoder_path = os.path.join(model_dir, "encoder.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        logger.info(f"Loading models from: {model_dir}")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)

        logger.info("✅ Models loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Features: {model.n_features_in_}")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model = encoder = scaler = None


def preprocess_property_data(property_data: PropertyData) -> pd.DataFrame:
    """Preprocess property data for prediction"""

    # Convert to dict and then DataFrame
    data_dict = {
        'municipality': property_data.municipality,
        'area': property_data.area,
        'number_of_rooms': property_data.number_of_rooms,
        'Балкон / Тераса': property_data.balkon_terasa,
        'Лифт': property_data.lift,
        'Приземје': property_data.prizemje,
        'Паркинг простор / Гаража': property_data.parking,
        'Поткровје': property_data.potkrovje,
        'Нова градба': property_data.nova_gradba,
        'Реновиран': property_data.renoviran,
        'Наместен': property_data.namesten,
        'Подрум': property_data.podrum,
        'Интерфон': property_data.interfon,
        'Дуплекс': property_data.dupleks
    }

    df = pd.DataFrame([data_dict])

    # Add date features
    current_date = datetime.now()
    df['year'] = current_date.year
    df['month'] = current_date.month
    df['weekday'] = current_date.weekday()

    # Define column groups
    categorical_cols = ['municipality']
    binary_cols = ['Балкон / Тераса', 'Лифт', 'Приземје', 'Паркинг простор / Гаража',
                   'Поткровје', 'Нова градба', 'Реновиран', 'Наместен', 'Подрум',
                   'Интерфон', 'Дуплекс']
    numerical_cols = ['area', 'number_of_rooms', 'year', 'month', 'weekday']

    # One-hot encode categorical features
    df_cat = pd.DataFrame(encoder.transform(df[categorical_cols]))
    df_cat.columns = encoder.get_feature_names_out(categorical_cols)

    # Scale numerical features
    df_num = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols)

    # Keep binary columns as is
    df_binary = df[binary_cols].reset_index(drop=True)

    # Combine processed features
    processed_data = pd.concat([df_num, df_cat, df_binary], axis=1)

    return processed_data


# Exception handler for better error messages
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


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Property Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
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
    Predict property price based on features

    Returns predicted price in EUR with price per square meter
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

        logger.info(
            f"Prediction: {predicted_price:.2f} EUR for {property_data.area}m² "
            f"in {property_data.municipality}"
        )

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            price_per_square_meter=round(price_per_square, 2),
            currency="EUR",
            success=True,
            model_info={
                "model_type": "RandomForestRegressor",
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
    """
    Predict prices for multiple properties

    Maximum 100 properties per request
    """

    if not all([model, encoder, scaler]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded properly"
        )

    results = []
    errors = []

    for i, property_data in enumerate(batch_data.properties):
        try:
            # Preprocess and predict
            processed_data = preprocess_property_data(property_data)
            predicted_price = model.predict(processed_data)[0]
            price_per_square = predicted_price / property_data.area

            results.append({
                "index": i,
                "predicted_price": round(predicted_price, 2),
                "price_per_square_meter": round(price_per_square, 2),
                "currency": "EUR",
                "municipality": property_data.municipality,
                "area": property_data.area
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
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""

    if not all([model, encoder, scaler]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    try:
        supported_municipalities = encoder.categories_[0].tolist()

        return ModelInfoResponse(
            model_type="RandomForestRegressor",
            n_estimators=getattr(model, 'n_estimators', None),
            max_depth=getattr(model, 'max_depth', None),
            n_features=getattr(model, 'n_features_in_', None),
            supported_municipalities=supported_municipalities,
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
    """Get list of supported municipalities"""

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