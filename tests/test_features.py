"""Тесты построения признаков. Не зависят от реальных данных: история создаётся здесь."""
import pandas as pd
import pytest

from src.features import (MIN_HISTORY, FeatureBuilder, NotEnoughHistoryError,
                          UnknownProductError, WeekOutOfRangeError)

FEATURE_NAMES = [
    "StockCode", "lag_1", "lag_2", "lag_4", "lag_8", "roll_mean_4",
    "roll_mean_8", "roll_max_4", "week_num", "month_num", "quarter_num",
    "weeks_since_start",
]


def make_history(skip_week_index=None):
    """Товар 'A': продажи 1, 2, 3, ... по неделям. Товар 'B': всегда 100."""
    weeks = list(pd.period_range("2011-01-03", periods=14, freq="W"))
    rows = []
    for i, w in enumerate(weeks):
        if i == skip_week_index:
            continue  # имитируем закрытую неделю: её нет в таблице
        rows.append(("A", w, i + 1))
        rows.append(("B", w, 100))
    return pd.DataFrame(rows, columns=["StockCode", "week", "Quantity"]), weeks


@pytest.fixture
def history_and_weeks():
    return make_history()


@pytest.fixture
def builder(history_and_weeks):
    history, _ = history_and_weeks
    return FeatureBuilder(history=history, feature_names=FEATURE_NAMES)


def test_values_match_hand_calculation(builder, history_and_weeks):
    _, weeks = history_and_weeks
    # целевая неделя с индексом 10: до неё продажи 1..10
    row = builder.build("A", weeks[10].start_time).iloc[0]
    assert row["lag_1"] == 10
    assert row["lag_2"] == 9
    assert row["lag_4"] == 7
    assert row["lag_8"] == 3
    assert row["roll_mean_4"] == pytest.approx((7 + 8 + 9 + 10) / 4)
    assert row["roll_mean_8"] == pytest.approx(sum(range(3, 11)) / 8)
    assert row["roll_max_4"] == 10
    assert row["weeks_since_start"] == 11
    # неделя 14-20 марта 2011
    assert row["week_num"] == 11
    assert row["month_num"] == 3
    assert row["quarter_num"] == 1


def test_columns_in_model_order(builder, history_and_weeks):
    _, weeks = history_and_weeks
    out = builder.build("A", weeks[10].start_time)
    assert list(out.columns) == FEATURE_NAMES
    assert len(out) == 1


def test_no_leakage_from_target_week(history_and_weeks):
    history, weeks = history_and_weeks
    target = weeks[10]
    normal = FeatureBuilder(history=history, feature_names=FEATURE_NAMES)

    changed = history.copy()
    changed.loc[(changed["StockCode"] == "A") & (changed["week"] == target), "Quantity"] = 99999
    leaky = FeatureBuilder(history=changed, feature_names=FEATURE_NAMES)

    # продажи самой целевой недели не должны влиять на признаки
    assert normal.build("A", target.start_time).equals(leaky.build("A", target.start_time))


def test_any_day_of_week_gives_same_result(builder, history_and_weeks):
    _, weeks = history_and_weeks
    monday = weeks[10].start_time
    sunday = weeks[10].end_time
    assert builder.build("A", monday).equals(builder.build("A", sunday))


def test_accepts_date_as_string(builder, history_and_weeks):
    _, weeks = history_and_weeks
    by_string = builder.build("A", "2011-03-16")  # среда той же недели
    assert by_string.equals(builder.build("A", weeks[10].start_time))


def test_lags_follow_rows_and_rank_is_dense():
    """Закрытая неделя отсутствует в таблице: лаги идут по строкам, ранг плотный."""
    history, weeks = make_history(skip_week_index=5)
    b = FeatureBuilder(history=history, feature_names=FEATURE_NAMES)
    row = b.build("A", weeks[10].start_time).iloc[0]
    # weeks[5] пропущена, значит перед weeks[10] идут строки 0,1,2,3,4,6,7,8,9
    # продажи (индекс+1): 1,2,3,4,5,7,8,9,10 -> lag_1 = 10, lag_8 = 2
    assert row["lag_1"] == 10
    assert row["lag_8"] == 2
    # неделя с индексом 10 в таблице стала 10-й (одну пропустили)
    assert row["weeks_since_start"] == 10


def test_unknown_product_raises(builder, history_and_weeks):
    _, weeks = history_and_weeks
    with pytest.raises(UnknownProductError):
        builder.build("NOPE", weeks[10].start_time)


def test_week_outside_history_raises(builder):
    with pytest.raises(WeekOutOfRangeError):
        builder.build("A", "2030-01-01")


def test_not_enough_history_raises(builder, history_and_weeks):
    _, weeks = history_and_weeks
    with pytest.raises(NotEnoughHistoryError):
        builder.build("A", weeks[MIN_HISTORY - 1].start_time)  # до неё только 7 недель
    # а на неделе с индексом MIN_HISTORY уже хватает
    builder.build("A", weeks[MIN_HISTORY].start_time)
