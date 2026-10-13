"""HTTP-сервис прогноза недельного спроса.

Запуск:  uvicorn src.api:app --reload
Документация (Swagger):  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.features import FeatureError, UnknownProductError
from src.predict import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # модель и история загружаются один раз при старте сервера
    app.state.predictor = Predictor()
    yield


app = FastAPI(
    title="Прогноз недельного спроса",
    description="Прогноз количества проданных единиц товара на неделю (модель CatBoost).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    # коды вроде 22423 можно присылать и числом, и строкой
    model_config = ConfigDict(coerce_numbers_to_str=True, str_strip_whitespace=True)

    stock_code: str = Field(min_length=1, examples=["85123A"])
    week: dt.date = Field(description="Любая дата внутри нужной недели (пн-вс)",
                          examples=["2011-11-14"])


class PredictResponse(BaseModel):
    stock_code: str
    week_start: dt.date
    week_end: dt.date
    predicted_quantity: float


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


@app.get("/health")
def health():
    """Проверка, что сервис жив и модель загружена."""
    return {"status": "ok"}


@app.get("/meta")
def meta(request: Request):
    """Какие товары и недели поддерживает сервис."""
    builder = get_predictor(request).builder
    weeks = builder.weeks
    return {
        "products": builder.products,
        "first_week": weeks[0].start_time.date(),
        "last_week": weeks[-1].start_time.date(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    """Прогноз продаж товара на неделю."""
    predictor = get_predictor(request)
    try:
        value = predictor.predict(body.stock_code, body.week)
    except UnknownProductError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FeatureError as e:
        raise HTTPException(status_code=422, detail=str(e))

    week = predictor.builder.to_week(body.week)
    return PredictResponse(
        stock_code=body.stock_code,
        week_start=week.start_time.date(),
        week_end=week.end_time.date(),
        predicted_quantity=round(value, 1),
    )
