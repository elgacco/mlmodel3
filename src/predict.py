"""Загрузка обученной модели и прогноз недельного спроса."""
from __future__ import annotations

from pathlib import Path

from catboost import CatBoostRegressor

from src.features import (MODEL_DIR, FeatureBuilder, load_feature_names,
                          load_history)


class Predictor:
    """Загружает модель и историю один раз и отвечает на запросы (товар, неделя)."""

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        model_dir = Path(model_dir)
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_dir / "model.cbm"))
        self.builder = FeatureBuilder(
            history=load_history(model_dir / "weekly_sales.csv"),
            feature_names=load_feature_names(model_dir / "meta.json"),
        )

    def predict(self, stock_code, week) -> float:
        """Прогноз продаж (штук) товара на указанную неделю.

        Ошибки признаков (FeatureError) пробрасываются наружу как есть.
        """
        features = self.builder.build(stock_code, week)
        pred = float(self.model.predict(features)[0])
        return max(0.0, pred)  # продажи не могут быть отрицательными
