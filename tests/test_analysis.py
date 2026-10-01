"""Independent checks of calendar alignment and observed-return inference.

Artificial observations below test software behavior; they are never study data.
"""
import json

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import grangercausalitytests

from src.analysis import build_design, granger, session_assign
from src import analysis, market


def test_publications_map_to_next_close_with_weekend_and_cutoff():
    sessions = pd.to_datetime(["2026-09-11", "2026-09-14", "2026-09-29"])
    times = pd.Series(pd.to_datetime([
        "2026-09-11T19:59:59Z",  # Friday, just before the close.
        "2026-09-11T20:00:00Z",  # Exact close belongs to this session.
        "2026-09-11T20:00:01Z",  # Friday after the close.
        "2026-09-12T12:00:00Z",  # Weekend.
        "2026-09-14T19:00:00Z",  # Monday before the close.
        "2026-09-29T20:00:00Z",  # Last included close.
        "2026-09-29T20:00:01Z",  # Beyond the last completed return.
        None,
    ], utc=True), index=list("abcdefgh"))
    expected = pd.Series(pd.to_datetime([
        "2026-09-11", "2026-09-11", "2026-09-14", "2026-09-14",
        "2026-09-14", "2026-09-29", None, None,
    ]), index=times.index)
    pd.testing.assert_series_equal(session_assign(times, sessions), expected)


def test_missing_sentiment_does_not_make_nonadjacent_sessions_neighbors():
    dates = pd.bdate_range("2026-09-08", periods=6)
    frame = pd.DataFrame({"return": [10, 11, 12, 13, 14, 15],
                          "tone": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6]}, index=dates)
    result = build_design(frame, "return", "tone", lag=1)
    # The missing current tone does not invalidate a return predicted yesterday;
    # the following session cannot borrow tone from two sessions earlier.
    assert dates[2] in result.index
    assert dates[3] not in result.index
    assert result.loc[dates[2], "x1"] == pytest.approx(0.2)
    assert result.loc[dates[4], "y1"] == 13
    frame.loc[dates[2], "return"] = np.nan
    assert dates[2] not in build_design(frame, "return", "tone").index
    assert dates[3] not in build_design(frame, "return", "tone").index


def test_change_and_forecast_lags_cost_two_sessions_without_preperiod_tone():
    dates = pd.bdate_range("2026-09-08", "2026-09-29")
    frame = pd.DataFrame({"return": np.arange(16.0),
                          "tone": np.linspace(-0.3, 0.3, 16)}, index=dates)
    frame["change"] = frame.tone.diff()
    one = build_design(frame, "return", "change", 1)
    two = build_design(frame, "return", "change", 2)
    assert len(one) == 14
    assert len(two) == 13
    assert one.index[0] == pd.Timestamp("2026-09-10")


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("effect", [-0.8, 0.8])
def test_granger_matches_independent_statsmodels_reference_and_effect_sign(lag, effect):
    rng = np.random.default_rng(984731)
    innovations = rng.normal(size=(320, 2))
    tone = np.zeros(320)
    returns = np.zeros(320)
    for t in range(1, len(tone)):
        tone[t] = 0.2 * tone[t - 1] + innovations[t, 0]
        returns[t] = 0.3 * returns[t - 1] + effect * tone[t - 1] + innovations[t, 1]
    frame = pd.DataFrame({"return": returns, "tone_change": tone},
                         index=pd.bdate_range("2025-01-02", periods=len(tone)))
    result = granger(frame, "return", "tone_change", lag, bootstrap=0)
    reference = grangercausalitytests(frame[["return", "tone_change"]],
                                     maxlag=[lag])[lag][0]["ssr_ftest"]
    assert result["status"] == "ok"
    assert result["f"] == pytest.approx(reference[0], rel=1e-10)
    assert result["p"] == pytest.approx(reference[1], rel=1e-9)
    assert result["df_resid"] == reference[2]
    assert result["coefficient_lag1"] * effect > 0
    assert result["ci_low_per_01"] < 0.1 * effect < result["ci_high_per_01"]
    assert result["effect_per_01"] == pytest.approx(result["coefficient_lag1"] * 0.1)


def test_unestimable_inputs_are_labeled_instead_of_reporting_false_significance():
    dates = pd.bdate_range("2026-09-08", periods=16)
    frame = pd.DataFrame({"return": np.arange(16.0), "tone_change": 0.0}, index=dates)
    assert granger(frame, "return", "tone_change", bootstrap=0)["status"] == "rank_deficient"
    frame.loc[dates[2:], "tone_change"] = np.nan
    result = granger(frame, "return", "tone_change", bootstrap=0)
    assert result["status"] == "insufficient_complete_rows"
    assert "p" not in result


def test_wild_bootstrap_is_reproducible_and_never_reports_zero_probability():
    rng = np.random.default_rng(716)
    frame = pd.DataFrame(rng.normal(size=(28, 2)), columns=["return", "tone_change"],
                         index=pd.bdate_range("2026-01-02", periods=28))
    first = granger(frame, "return", "tone_change", bootstrap=199)
    second = granger(frame, "return", "tone_change", bootstrap=199)
    assert first["wild_p"] == second["wild_p"]
    assert 1 / 200 <= first["wild_p"] <= 1
    assert first["bootstrap_draws"] == 199


def _mock_market_download(monkeypatch, tmp_path, *, missing_constituent=False, missing_baseline=False,
                          missing_entire_session=False,
                          observation_time="2026-09-11 18:00"):
    """Create fake Yahoo responses, retaining the real acquisition/analysis path."""
    monkeypatch.setattr(market, "ROOT", tmp_path)
    monkeypatch.setattr(market, "CUTOFF", "2026-09-11")
    monkeypatch.setattr(market, "validate_cutoff", lambda: pd.Timestamp(
        observation_time, tz="America/New_York"))
    dates = pd.to_datetime(["2026-09-04", "2026-09-08", "2026-09-09",
                            "2026-09-10", "2026-09-11"])
    times = (dates.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30))
    seconds = [int(d.timestamp()) for d in times]

    def get(url, **kwargs):
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        if missing_constituent and url.endswith("/NVDA"):
            prices[2] = None
        if missing_baseline and url.endswith("/NVDA"):
            prices[0] = None
        response_times = seconds
        if missing_entire_session:
            prices = [v for i, v in enumerate(prices) if i != 2]
            response_times = [v for i, v in enumerate(seconds) if i != 2]
        content = json.dumps({"chart": {"result": [{
            "timestamp": response_times,
            "indicators": {"adjclose": [{"adjclose": prices}],
                           "quote": [{"close": prices}]},
        }]}}).encode()

        class Response:
            def raise_for_status(self):
                pass

        response = Response()
        response.content, response.url = content, url
        return response

    monkeypatch.setattr(market.requests, "get", get)
    return market.collect()


def test_market_drops_intraday_prices_before_calculating_completed_returns(monkeypatch, tmp_path):
    daily = _mock_market_download(monkeypatch, tmp_path,
                                  observation_time="2026-09-11 15:59")
    assert daily.index[-1] == pd.Timestamp("2026-09-10")
    assert daily.loc["2026-09-08", "basket"] == pytest.approx(1.0)
    assert daily.loc["2026-09-10", "basket_wealth"] == pytest.approx(1.03)
    assert daily.loc["2026-09-10", "buy_hold_wealth"] == pytest.approx(1.03)


def test_missing_constituent_cannot_reweight_buy_hold_or_resume_unknown_wealth(monkeypatch, tmp_path):
    daily = _mock_market_download(monkeypatch, tmp_path, missing_constituent=True)
    assert pd.isna(daily.loc["2026-09-09", "basket"])
    assert pd.isna(daily.loc["2026-09-10", "basket"])
    assert pd.isna(daily.loc["2026-09-09", "buy_hold"])
    assert pd.isna(daily.loc["2026-09-10", "buy_hold"])
    assert np.isfinite(daily.loc["2026-09-11", "basket"])
    assert pd.isna(daily.loc["2026-09-11", "basket_wealth"])
    assert pd.isna(daily.loc["2026-09-11", "buy_hold_wealth"])


def test_buy_hold_cannot_invent_initial_shares_without_a_baseline_price(monkeypatch, tmp_path):
    daily = _mock_market_download(monkeypatch, tmp_path, missing_baseline=True)
    assert daily.buy_hold.isna().all()
    assert daily.buy_hold_wealth.isna().all()
    # Subsequent one-session market returns remain observable independently.
    assert np.isfinite(daily.loc["2026-09-11", "basket"])


def test_session_missing_from_every_feed_is_retained_as_a_calendar_gap(monkeypatch, tmp_path):
    daily = _mock_market_download(monkeypatch, tmp_path, missing_entire_session=True)
    assert daily.index.equals(pd.bdate_range("2026-09-08", "2026-09-11", name="date"))
    assert daily.loc["2026-09-09", ["basket", "spy", "qqq", "buy_hold"]].isna().all()
    assert daily.loc["2026-09-10", ["basket", "spy", "qqq", "buy_hold"]].isna().all()
    assert daily.loc["2026-09-11", ["basket", "spy", "qqq", "buy_hold"]].notna().all()
    assert daily.loc["2026-09-11", ["basket_wealth", "spy_wealth", "qqq_wealth", "buy_hold_wealth"]].isna().all()


def test_analysis_declines_adf_when_sentiment_has_an_internal_missing_session(monkeypatch, tmp_path):
    """Exercise diagnostics through real analysis outputs, without a live corpus."""
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "CUTOFF", "2026-09-29")
    monkeypatch.setattr(analysis, "bias_analysis", lambda data, out: None)
    real_granger = analysis.granger
    monkeypatch.setattr(analysis, "granger", lambda frame, y, x, lag=1:
                        real_granger(frame, y, x, lag, bootstrap=0))
    folder = tmp_path / "data" / "processed"
    folder.mkdir(parents=True)
    dates = pd.bdate_range("2026-09-08", "2026-09-29")
    rng = np.random.default_rng(90731)
    returns = pd.DataFrame(rng.normal(size=(16, 4)), index=dates,
                           columns=["basket", "spy", "qqq", "buy_hold"])
    returns["spread"] = returns.basket - returns.spy
    returns["spread_qqq"] = returns.basket - returns.qqq
    for col in ["basket", "spy", "qqq", "buy_hold"]:
        returns[col + "_wealth"] = (1 + returns[col] / 100).cumprod()
    returns.to_csv(folder / "market_daily.csv", index_label="date")
    level_dates = pd.DatetimeIndex([pd.Timestamp("2026-09-04"), *dates])
    level_values = np.r_[100.0, 100 * returns.basket_wealth.to_numpy()]
    pd.concat([pd.DataFrame({"date": level_dates, "ticker": ticker,
                             "adjusted_close": level_values})
               for ticker in market.TICKERS], ignore_index=True).to_csv(
                   folder / "market_levels.csv", index=False)
    records = []
    for date in dates:
        for medium in ["news", "comments", "stories"]:
            if medium == "news" and date == dates[6]:
                continue
            value = float(rng.uniform(-0.8, 0.8))
            records.append({
                "included_analysis": True, "date": date.strftime("%Y-%m-%d"),
                "published_at": date.strftime("%Y-%m-%dT19:00:00Z"),
                "kind": "news" if medium == "news" else "social",
                "is_comment": medium == "comments", "risk": True,
                "vader": value, "negative": float(value < -0.05),
                "hu_negative": abs(value) / 10, "thread_id": "test_thread",
                "author": "test_author", "source": "test_source",
                "included_strict_sensitivity": True,
            })
    pd.DataFrame(records).to_csv(folder / "scored_documents.csv", index=False)
    analysis.analyze()
    diagnostics = pd.read_csv(tmp_path / "outputs" / "stationarity.csv").set_index("variable")
    for variable in ["balanced_risk", "d_balanced_risk"]:
        assert diagnostics.loc[variable, "status"] == "skipped_internal_gaps"
        assert pd.isna(diagnostics.loc[variable, "p"])
    assert diagnostics.loc["basket", "status"] == "ok"
    assert np.isfinite(diagnostics.loc["basket", "p"])
