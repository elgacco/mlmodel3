"""Построение признаков для прогноза недельного спроса.

Признаки считаются ровно так же, как в ноутбуке retail.ipynb:
  - лаги и скользящие считаются по СТРОКАМ товара (а не по календарю);
  - используются только недели строго ДО целевой недели;
  - календарные признаки берутся из недельного Period (неделя пн-вс).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Минимум прошлых недель: нужен lag_8 и roll_mean_8
MIN_HISTORY = 8


class FeatureError(ValueError):
    """Базовая ошибка построения признаков."""


class UnknownProductError(FeatureError):
    """Товара нет в истории (не входит в 288)."""


class WeekOutOfRangeError(FeatureError):
    """Целевой недели нет в истории."""


class NotEnoughHistoryError(FeatureError):
    """До целевой недели накопилось меньше MIN_HISTORY недель."""


def load_history(path: Path | str = MODEL_DIR / "weekly_sales.csv") -> pd.DataFrame:
    """Читает недельную таблицу продаж: колонки StockCode, week (Period), Quantity."""
    df = pd.read_csv(path, usecols=["StockCode", "Quantity", "week_start"],
                     dtype={"StockCode": str})
    df["week"] = pd.to_datetime(df["week_start"]).dt.to_period("W")
    df = df.drop(columns="week_start")
    return df.sort_values(["StockCode", "week"]).reset_index(drop=True)


def load_feature_names(path: Path | str = MODEL_DIR / "meta.json") -> list[str]:
    """Порядок колонок, в котором признаки подавались в модель при обучении."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


class FeatureBuilder:
    """Хранит историю в памяти и строит признаки для пары (товар, неделя)."""

    def __init__(self, history: pd.DataFrame | None = None,
                 feature_names: list[str] | None = None):
        history = load_history() if history is None else history
        self.feature_names = feature_names or load_feature_names()

        # история каждого товара: Series продаж, отсортированная по неделям
        self._by_product = {
            code: grp.set_index("week")["Quantity"].sort_index()
            for code, grp in history.groupby("StockCode")
        }

        # weeks_since_start: плотный ранг недели среди всех недель таблицы
        # (как rank(method='dense') в ноутбуке; закрытая неделя 27.12.2010
        # в таблице отсутствует, поэтому нумерация её пропускает)
        all_weeks = sorted(history["week"].unique())
        self._week_rank = {w: i + 1 for i, w in enumerate(all_weeks)}

    @property
    def products(self) -> list[str]:
        return sorted(self._by_product)

    @property
    def weeks(self) -> list[pd.Period]:
        return sorted(self._week_rank)

    @staticmethod
    def to_week(week) -> pd.Period:
        """Любая дата внутри недели (str, date, Timestamp, Period) -> недельный Period."""
        return pd.Period(week, freq="W")

    def build(self, stock_code, week) -> pd.DataFrame:
        """Признаки одной строки: DataFrame из 1 строки в порядке feature_names."""
        code = str(stock_code)
        if code not in self._by_product:
            raise UnknownProductError(f"Товар {code!r} не входит в список товаров модели")

        target = self.to_week(week)
        if target not in self._week_rank:
            raise WeekOutOfRangeError(f"Недели {target} нет в истории")

        # только прошлое: строго до целевой недели
        sales = self._by_product[code]
        past = sales[sales.index < target]
        if len(past) < MIN_HISTORY:
            raise NotEnoughHistoryError(
                f"Для товара {code!r} до недели {target} только {len(past)} недель "
                f"истории, нужно минимум {MIN_HISTORY}"
            )

        last = past.to_numpy()  # от старых к новым
        feats = {
            "StockCode": code,
            "lag_1": last[-1],
            "lag_2": last[-2],
            "lag_4": last[-4],
            "lag_8": last[-8],
            "roll_mean_4": last[-4:].mean(),
            "roll_mean_8": last[-8:].mean(),
            "roll_max_4": last[-4:].max(),
            "week_num": target.week,
            "month_num": target.month,
            "quarter_num": target.quarter,
            "weeks_since_start": self._week_rank[target],
        }
        return pd.DataFrame([feats])[self.feature_names]
