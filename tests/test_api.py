"""Тесты HTTP-сервиса. Используют реальные файлы из папки model/."""
import pytest
from fastapi.testclient import TestClient

from src.features import MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.cbm").exists(),
    reason="В папке model/ нет файла модели",
)


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:  # with -> выполняется загрузка модели при старте
        yield c


@pytest.fixture(scope="module")
def sample(client):
    meta = client.get("/meta").json()
    return {"code": meta["products"][0], "week": meta["last_week"], "meta": meta}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_meta(client, sample):
    meta = sample["meta"]
    assert len(meta["products"]) > 0
    assert meta["first_week"] < meta["last_week"]


def test_predict_ok(client, sample):
    r = client.post("/predict", json={"stock_code": sample["code"], "week": sample["week"]})
    assert r.status_code == 200
    body = r.json()
    assert body["stock_code"] == sample["code"]
    assert body["predicted_quantity"] >= 0
    assert body["week_start"] <= sample["week"] <= body["week_end"]


def test_predict_accepts_any_day_in_week(client, sample):
    r1 = client.post("/predict", json={"stock_code": sample["code"], "week": sample["week"]})
    r2 = client.post("/predict", json={"stock_code": sample["code"], "week": r1.json()["week_end"]})
    assert r1.json() == r2.json()


def test_unknown_product_404(client, sample):
    r = client.post("/predict", json={"stock_code": "NOPE", "week": sample["week"]})
    assert r.status_code == 404
    assert "NOPE" in r.json()["detail"]


def test_week_outside_history_422(client, sample):
    r = client.post("/predict", json={"stock_code": sample["code"], "week": "2030-01-01"})
    assert r.status_code == 422


def test_not_enough_history_422(client, sample):
    first = sample["meta"]["first_week"]
    r = client.post("/predict", json={"stock_code": sample["code"], "week": first})
    assert r.status_code == 422


def test_bad_date_format_422(client, sample):
    r = client.post("/predict", json={"stock_code": sample["code"], "week": "не дата"})
    assert r.status_code == 422


def test_missing_field_422(client):
    r = client.post("/predict", json={"stock_code": "85123A"})
    assert r.status_code == 422
