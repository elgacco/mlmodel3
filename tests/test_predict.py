"""Тесты прогноза. Используют реальные файлы из папки model/."""
import math

import pytest

from src.features import MODEL_DIR, UnknownProductError

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.cbm").exists(),
    reason="В папке model/ нет файла модели",
)


@pytest.fixture(scope="module")
def predictor():
    from src.predict import Predictor
    return Predictor()


def test_prediction_is_nonnegative_finite_float(predictor):
    code = predictor.builder.products[0]
    week = predictor.builder.weeks[-1].start_time
    value = predictor.predict(code, week)
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value >= 0


def test_all_products_predictable_on_late_week(predictor):
    week = predictor.builder.weeks[-2].start_time
    for code in predictor.builder.products:
        value = predictor.predict(code, week)
        assert math.isfinite(value) and value >= 0


def test_unknown_product_raises(predictor):
    week = predictor.builder.weeks[-1].start_time
    with pytest.raises(UnknownProductError):
        predictor.predict("NOT_A_PRODUCT", week)
