import gzip
import json
import math
import sys
import random
from pathlib import Path
from preview import run_regime, DATA

TEMPLATE_ZAID = """
from __future__ import annotations
from typing import Any

LARGE_CAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO",
    "MU", "QCOM", "MRVL", "AMAT", "LRCX",
    "JPM", "V", "MA", "GS", "MS",
    "UNH", "LLY", "ABBV", "JNJ",
    "XOM", "CVX",
    "COST", "HD", "NFLX", "CRM", "PLTR",
]

BENCHMARK = "SPY"
MIN_PRICE = 10.0
MIN_BARS = 70

# Tunable Knobs
TREND_WINDOW = __TREND_WINDOW__
STOCK_TREND_WIN = __STOCK_TREND_WIN__
MOM_LONG = __MOM_LONG__
MOM_SHORT = __MOM_SHORT__
MOM_LONG_WT = 0.50
MOM_SHORT_WT = 0.50
TOP_N = __TOP_N__
MAX_POSITION = __MAX_POSITION__
GROSS_CAP = __GROSS_CAP__
REBAL_THRESHOLD = __REBAL_THRESHOLD__
TREND_BAND = __TREND_BAND__

_last_risk_on = False

def _closes(bars: list[dict]) -> list[float]:
    return [float(b["close"]) for b in bars]

def _sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def _momentum(prices: list[float], lookback: int) -> float | None:
    if len(prices) < lookback + 1:
        return None
    base = prices[-(lookback + 1)]
    if base <= 0:
        return None
    return (prices[-1] / base) - 1.0

def _equity(portfolio_state: dict, cash: float) -> float:
    last_prices = portfolio_state.get("last_prices", {})
    positions = portfolio_state.get("positions", [])
    holdings_val = sum(
        p["quantity"] * last_prices.get(p["ticker"], p.get("avg_cost", 0))
        for p in positions
    )
    return cash + holdings_val

def _current_holdings(portfolio_state: dict) -> dict[str, int]:
    return {p["ticker"]: int(p["quantity"]) for p in portfolio_state.get("positions", [])}

def _get_price(ticker: str, portfolio_state: dict, market_state: dict) -> float | None:
    price = portfolio_state.get("last_prices", {}).get(ticker)
    if not price or price <= 0:
        bars = market_state.get(ticker, [])
        price = bars[-1]["close"] if bars else None
    return float(price) if price and price > 0 else None

def _liquidate_all(holdings: dict[str, int]) -> list[dict]:
    return [
        {"ticker": t, "side": "sell", "quantity": q}
        for t, q in holdings.items()
        if q > 0
    ]

def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    global _last_risk_on
    spy_bars = market_state.get(BENCHMARK, [])
    if len(spy_bars) < MIN_BARS:
        return []

    spy_closes = _closes(spy_bars)
    holdings = _current_holdings(portfolio_state)
    equity = _equity(portfolio_state, cash)

    spy_sma = _sma(spy_closes, TREND_WINDOW)
    if spy_sma is not None:
        strong_on = spy_closes[-1] > spy_sma * (1 + TREND_BAND)
        clearly_off = spy_closes[-1] < spy_sma * (1 - TREND_BAND)
        if _last_risk_on:
            risk_on = not clearly_off
        else:
            risk_on = strong_on
    else:
        risk_on = False
        
    _last_risk_on = risk_on

    if not risk_on:
        return _liquidate_all(holdings)

    scores: dict[str, float] = {}

    for ticker in LARGE_CAPS:
        bars = market_state.get(ticker, [])
        if not bars or len(bars) < MOM_LONG + 2:
            continue

        closes_series = _closes(bars)
        price = closes_series[-1]

        if price < MIN_PRICE:
            continue

        stock_sma = _sma(closes_series, STOCK_TREND_WIN)
        if stock_sma is None or price < stock_sma:
            continue

        mom_long = _momentum(closes_series, MOM_LONG)
        mom_short = _momentum(closes_series, MOM_SHORT)

        if mom_long is None or mom_short is None:
            continue

        blend = MOM_LONG_WT * mom_long + MOM_SHORT_WT * mom_short

        if blend > 0:
            scores[ticker] = blend

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winners = [t for t, _ in ranked[:TOP_N]]

    if not winners:
        return _liquidate_all(holdings)

    n = len(winners)
    raw_weight = min(1.0 / n, MAX_POSITION)

    total_gross = raw_weight * n
    if total_gross > GROSS_CAP:
        raw_weight = GROSS_CAP / n

    targets = {ticker: raw_weight for ticker in winners}

    orders: list[dict] = []

    for ticker, qty in holdings.items():
        if ticker not in winners and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})

    for ticker, weight in targets.items():
        price = _get_price(ticker, portfolio_state, market_state)
        if not price:
            continue

        target_qty = int((equity * weight) / price)
        current_qty = holdings.get(ticker, 0)
        diff = target_qty - current_qty

        current_weight = (current_qty * price) / equity if equity > 0 else 0
        drift = abs(current_weight - weight)

        if diff > 0 and (current_qty == 0 or drift > REBAL_THRESHOLD):
            orders.append({"ticker": ticker, "side": "buy", "quantity": diff})
        elif diff < 0 and drift > REBAL_THRESHOLD:
            orders.append({"ticker": ticker, "side": "sell", "quantity": abs(diff)})

    return orders
"""

def generate_params():
    # Tunable around Zaid's parameters
    return {
        "trend_window": random.choice([40, 50, 60, 75]),
        "stock_trend_win": random.choice([40, 50, 60, 75]),
        "mom_long": random.choice([50, 63, 75]),
        "mom_short": random.choice([15, 21, 30]),
        "top_n": random.choice([4, 5, 6]),
        "max_position": random.choice([0.24, 0.28, 0.30]),
        "gross_cap": random.choice([1.30, 1.40, 1.45]),
        "rebal_threshold": random.choice([0.02, 0.03, 0.04]),
        "trend_band": random.choice([0.00, 0.005, 0.01, 0.015, 0.02])
    }

def main():
    regimes = json.loads(gzip.open(DATA, "rb").read())
    here = Path(__file__).parent
    temp_path = here / "temp_agent.py"
    
    random.seed(42)
    attempts = 200
    
    print(f"Sweeping {attempts} variations of Zaid's strategy with trend band...")
    
    matching = []
    
    for i in range(attempts):
        params = generate_params()
        
        code = TEMPLATE_ZAID
        for k, v in params.items():
            placeholder = f"__{k.upper()}__"
            code = code.replace(placeholder, str(v))
            
        with open(temp_path, "w") as f:
            f.write(code)
            
        try:
            results = []
            for name, reg in regimes.items():
                res = run_regime(temp_path, name, reg)
                results.append(res)
                
            res_dict = {r["name"]: r for r in results}
            uptrend_calmar = res_dict["calm_uptrend"]["calmar"]
            selloff_calmar = res_dict["moderate_selloff"]["calmar"]
            volspike_calmar = res_dict["vol_spike_snapback"]["calmar"]
            
            peak_gross = max(r["peak_gross"] for r in results)
            worst_streak = max(r["max_conc_streak"] for r in results)
            worst_dd = max(r["mdd"] for r in results)
            
            passed = peak_gross <= 1.500001 and worst_streak <= 5 and worst_dd < 0.50
            
            # Check if it beats Zaid on ALL THREE:
            # Zaid's stats: Uptrend 23.80, Selloff -5.77, VolSpike -2.80
            beats_all = (
                passed
                and uptrend_calmar >= 23.80
                and selloff_calmar >= -5.77
                and volspike_calmar >= -2.80
            )
            
            avg_calmar = sum(r["calmar"] for r in results) / len(results)
            
            if beats_all:
                matching.append((avg_calmar, params, results))
                print(f"Match found beating Zaid in ALL three: Avg Calmar = {avg_calmar:.4f}")
        except Exception as e:
            pass
            
    if temp_path.exists():
        temp_path.unlink()
        
    print("\n" + "="*80)
    if matching:
        matching.sort(key=lambda x: x[0], reverse=True)
        best = matching[0]
        print("OPTIMIZED ZAID AGENT BEATING BENCHMARK IN ALL THREE SECTIONS:")
        print(json.dumps(best[1], indent=2))
        print("\nRESULTS:")
        for r in best[2]:
            print(f"  {r['name']:20s} {r['ret']*100:6.2f}% {r['mdd']*100:6.2f}% "
                  f"{r['sharpe']:7.2f} {r['calmar']:7.2f} {r['trades']:7d}")
        print(f"  AVERAGE CALMAR: {best[0]:.4f}")
    else:
        print("NO PARAMETER COMBINATION BEAT THE BENCHMARK IN ALL THREE REGIMES SIMULTANEOUSLY.")
    print("="*80)

if __name__ == "__main__":
    main()
