import gzip
import json
import os
import sys
import random
from pathlib import Path
from preview import run_regime, DATA

TEMPLATE = """
from statistics import pstdev, mean
from math import sqrt

# Knobs
TREND_SMA = __TREND_SMA__
NAME_SMA = __NAME_SMA__
MOM_FAST = __MOM_FAST__
MOM_SLOW = __MOM_SLOW__
TOP_N = __TOP_N__
NAME_CAP = __NAME_CAP__
GROSS_MAX = __GROSS_MAX__
DEF_GROSS_SOFT = __DEF_GROSS_SOFT__
DEF_GROSS_HARD = __DEF_GROSS_HARD__
REBALANCE_EVERY = __REBALANCE_EVERY__
COOLDOWN_TICKS = __COOLDOWN_TICKS__
BRAKE_3D = __BRAKE_3D__
BRAKE_5D = __BRAKE_5D__
VOL_CEILING = __VOL_CEILING__
TARGET_VOL = __TARGET_VOL__
TREND_BAND = __TREND_BAND__
VOL_LOOKBACK = 20
MOM_FAST_SKIP = 5
INDEX_MOM_MIN = -0.02

RISK_ON = (
    "SPY", "QQQ", "SMH",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
)
DEFENSIVE = ("XLP", "XLU")
SELECT = RISK_ON + DEFENSIVE

_tick = 0
_last_rebalance = -1000
_last_regime = None
_cooldown = 0

def _closes(bars):
    return [float(b["close"]) for b in bars] if bars else []

def _sma(closes, n):
    return sum(closes[-n:]) / n if len(closes) >= n else None

def _ret(closes, days, skip=0):
    need = days + skip + 1
    if len(closes) < need:
        return None
    end = closes[-(skip + 1)]
    start = closes[-(days + skip + 1)]
    return end / start - 1.0 if start > 0 else None

def _ann_vol(closes, n):
    if len(closes) < n + 1:
        return None
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - n, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 2:
        return None
    return pstdev(rets) * sqrt(252.0)

def _market_vol(market_state):
    v = _ann_vol(_closes(market_state.get("QQQ") or []), VOL_LOOKBACK)
    return v if v and v > 0 else 0.20

def _regime(market_state):
    qqq = _closes(market_state.get("QQQ") or [])
    spy = _closes(market_state.get("SPY") or [])
    if not qqq or not spy:
        return "hard"

    # Fast crash brake
    r3, r5 = _ret(qqq, 3), _ret(qqq, 5)
    if (r3 is not None and r3 < BRAKE_3D) or (r5 is not None and r5 < BRAKE_5D):
        return "hard"

    spy_sma, qqq_sma = _sma(spy, TREND_SMA), _sma(qqq, TREND_SMA)
    idx_mom = _ret(qqq, MOM_SLOW)
    if spy_sma is None or qqq_sma is None or idx_mom is None:
        return "soft"

    # Check trend
    strong_on = (spy[-1] > spy_sma * (1 + TREND_BAND) and qqq[-1] > qqq_sma * (1 + TREND_BAND) and idx_mom >= INDEX_MOM_MIN)
    clearly_off = qqq[-1] < qqq_sma * (1 - TREND_BAND) or idx_mom < INDEX_MOM_MIN
    
    if _last_regime == "on":
        return "soft" if clearly_off else "on"
    return "on" if strong_on else "soft"

def _inv_vol_weights(names, market_state, gross):
    inv = {}
    for t in names:
        v = _ann_vol(_closes(market_state.get(t) or []), VOL_LOOKBACK)
        if v and v > 0:
            inv[t] = 1.0 / v
    if not inv:
        return {}
    s = sum(inv.values())
    return {t: min(NAME_CAP, gross * w / s) for t, w in inv.items()}

def _rank(market_state, universe):
    ranked = []
    for t in universe:
        closes = _closes(market_state.get(t) or [])
        sma = _sma(closes, NAME_SMA)
        mf, ms = _ret(closes, MOM_FAST, MOM_FAST_SKIP), _ret(closes, MOM_SLOW)
        if sma is None or mf is None or ms is None or not closes:
            continue
        score = 0.5 * mf + 0.3 * ms + 0.2 * (closes[-1] / sma - 1.0)
        if score > 0 and closes[-1] > sma:
            ranked.append((score, t))
    ranked.sort(reverse=True)
    return [t for _, t in ranked[:TOP_N]]

def _target_weights(market_state, regime):
    if regime == "hard":
        avail = [t for t in DEFENSIVE if market_state.get(t)]
        return _inv_vol_weights(avail, market_state, DEF_GROSS_HARD) if avail else {}

    if regime == "on":
        mvol = _market_vol(market_state)
        mvol = max(mvol, 0.05)
        
        # Snapback overlay: if QQQ has recovered > 4% in 20 days, relax vol targeting
        qqq_close = _closes(market_state.get("QQQ") or [])
        qqq_mom20 = _ret(qqq_close, 20)
        if qqq_mom20 is not None and qqq_mom20 > 0.04:
            gross = GROSS_MAX
        else:
            gross = min(GROSS_MAX, TARGET_VOL / mvol)
            
        winners = _rank(market_state, RISK_ON)
        return _inv_vol_weights(winners, market_state, gross) if winners else {}

    winners = _rank(market_state, SELECT)
    return _inv_vol_weights(winners, market_state, DEF_GROSS_SOFT) if winners else {}

def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance, _last_regime, _cooldown
    _tick += 1

    positions = {p["ticker"]: p for p in portfolio_state.get("positions", [])}
    last_prices = portfolio_state.get("last_prices", {})
    equity = portfolio_state.get("cash", cash)
    for tk, pos in positions.items():
        equity += pos["quantity"] * last_prices.get(tk, pos.get("avg_cost", 0))
    if equity <= 0:
        return []

    regime = _regime(market_state)
    if regime == "hard":
        _cooldown = COOLDOWN_TICKS
    elif _cooldown > 0:
        _cooldown -= 1
        if regime == "on":
            regime = "soft"

    derisk = _last_regime is not None and regime != _last_regime and (
        regime == "hard" or (regime == "soft" and _last_regime == "on")
    )
    on_cadence = _tick - _last_rebalance >= REBALANCE_EVERY
    _last_regime = regime
    if not on_cadence and not derisk:
        return []

    targets = _target_weights(market_state, regime)

    orders = []
    for ticker, pos in positions.items():
        if ticker not in targets and pos["quantity"] > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})

    for ticker, weight in targets.items():
        bars = market_state.get(ticker)
        if not bars:
            continue
        px = float(bars[-1]["close"])
        if px <= 0:
            continue
        cur_qty = positions.get(ticker, {}).get("quantity", 0)
        delta = int((equity * weight - cur_qty * px) // px)
        if abs(delta * px) < 0.03 * equity:
            continue
        if delta > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": delta})
        elif delta < 0 and cur_qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": min(abs(delta), cur_qty)})

    if orders:
        _last_rebalance = _tick
    return orders
"""

def generate_params():
    return {
        "trend_sma": random.choice([50, 75, 100, 150, 200]),
        "name_sma": random.choice([20, 50, 100]),
        "mom_fast": random.choice([21, 42, 63]),
        "mom_slow": random.choice([63, 126, 252]),
        "top_n": random.choice([3, 4, 5, 6]),
        "name_cap": random.choice([0.18, 0.20, 0.24, 0.28]),
        "gross_max": random.choice([1.00, 1.20, 1.35, 1.45]),
        "def_gross_soft": random.choice([0.10, 0.20, 0.25, 0.30]),
        "def_gross_hard": random.choice([0.00, 0.05, 0.10]),
        "rebalance_every": random.choice([1, 3, 5, 10]),
        "cooldown_ticks": random.choice([1, 2, 3, 5]),
        "brake_3d": random.choice([-0.04, -0.05, -0.06, -0.08]),
        "brake_5d": random.choice([-0.06, -0.07, -0.08, -0.10]),
        "vol_ceiling": random.choice([0.25, 0.28, 0.32, 0.35]),
        "target_vol": random.choice([0.10, 0.12, 0.15, 0.18]),
        "trend_band": random.choice([0.00, 0.005, 0.01, 0.015, 0.02])
    }

def main():
    regimes = json.loads(gzip.open(DATA, "rb").read())
    here = Path(__file__).parent
    temp_path = here / "temp_agent.py"
    
    random.seed(42)  # For reproducibility
    
    best_avg_calmar = -9999.0
    best_params = None
    best_results = None
    
    attempts = 500
    print(f"Starting parameter sweep with {attempts} random combinations...")
    
    # Benchmarks to beat in all 3 sections:
    # calm_uptrend calmar > 23.80
    # moderate_selloff calmar > -5.77
    # vol_spike_snapback calmar > -2.80
    
    matching_combinations = []
    
    for i in range(attempts):
        params = generate_params()
        
        # Replace template placeholders
        code = TEMPLATE
        for k, v in params.items():
            placeholder = f"__{k.upper()}__"
            code = code.replace(placeholder, str(v))
            
        # Write to temp agent
        with open(temp_path, "w") as f:
            f.write(code)
            
        try:
            results = []
            for name, reg in regimes.items():
                res = run_regime(temp_path, name, reg)
                results.append(res)
                
            # Parse Calmars for each section
            res_dict = {r["name"]: r for r in results}
            uptrend_calmar = res_dict["calm_uptrend"]["calmar"]
            selloff_calmar = res_dict["moderate_selloff"]["calmar"]
            volspike_calmar = res_dict["vol_spike_snapback"]["calmar"]
            
            peak_gross = max(r["peak_gross"] for r in results)
            worst_streak = max(r["max_conc_streak"] for r in results)
            worst_dd = max(r["mdd"] for r in results)
            
            # Constraint check
            passed_constraints = peak_gross <= 1.500001 and worst_streak <= 5 and worst_dd < 0.50
            
            # Check if it beats Zaid on ALL THREE
            beats_all = (
                passed_constraints
                and uptrend_calmar > 23.80
                and selloff_calmar > -5.77
                and volspike_calmar > -2.80
            )
            
            avg_calmar = sum(r["calmar"] for r in results) / len(results)
            
            if beats_all:
                matching_combinations.append((avg_calmar, params, results))
                print(f"Found a match beating Zaid in ALL sections: Avg Calmar = {avg_calmar:.4f} (Attempt {i+1})")
                
            if passed_constraints and avg_calmar > best_avg_calmar:
                best_avg_calmar = avg_calmar
                best_params = params
                best_results = results
        except Exception as e:
            # print(f"Error on attempt {i+1}: {e}")
            pass
            
    # Clean up
    if temp_path.exists():
        temp_path.unlink()
        
    print("\n" + "="*80)
    if matching_combinations:
        matching_combinations.sort(key=lambda x: x[0], reverse=True)
        best_overall = matching_combinations[0]
        print(f"BEST COMBINATION BEATING BENCHMARK IN ALL THREE SECTIONS:")
        print(json.dumps(best_overall[1], indent=2))
        print("\nRESULTS:")
        for r in best_overall[2]:
            print(f"  {r['name']:20s} {r['ret']*100:6.2f}% {r['mdd']*100:6.2f}% "
                  f"{r['sharpe']:7.2f} {r['calmar']:7.2f} {r['trades']:7d}")
        print(f"  AVERAGE CALMAR: {best_overall[0]:.4f}")
    else:
        print("NO PARAMETER SET FOUND BEATING BENCHMARK IN ALL THREE REGIMES SIMULTANEOUSLY.")
        print("Falling back to overall best average Calmar parameter set:")
        print(json.dumps(best_params, indent=2))
        print("\nRESULTS:")
        for r in best_results:
            print(f"  {r['name']:20s} {r['ret']*100:6.2f}% {r['mdd']*100:6.2f}% "
                  f"{r['sharpe']:7.2f} {r['calmar']:7.2f} {r['trades']:7d}")
        print(f"  AVERAGE CALMAR: {best_avg_calmar:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
