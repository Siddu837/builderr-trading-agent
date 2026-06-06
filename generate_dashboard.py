import gzip
import json
import math
import sys
from pathlib import Path
from statistics import pstdev

HERE = Path(__file__).parent
DATA = HERE / "sample_regimes.json.gz"
BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}
SLIP_EQUITY = 0.0005
SLIP_LEVERAGED = 0.0010
START_CASH = 100_000.0

def beta(ticker: str) -> float:
    if ticker in BETA_3X:
        return 3.0
    if ticker in BETA_2X:
        return 2.0
    return 1.0

def load_decide(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide

def _expand(rows):
    return [
        {"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in rows
    ]

def run_regime_detailed(agent_path: Path, name: str, regime: dict) -> dict:
    decide = load_decide(agent_path)
    bars = {t: _expand(rows) for t, rows in regime["bars"].items()}
    all_dates = sorted({b["ts"] for rows in bars.values() for b in rows})
    eval_dates = [d for d in all_dates if regime["eval_start"] <= d <= regime["eval_end"]]

    cash = START_CASH
    positions = {}
    avg_cost = {}
    equity_curve = []
    dates = []
    trades = 0
    pending = []

    def close_on(ticker, date):
        for b in bars.get(ticker, []):
            if b["ts"] == date:
                return b["close"]
        return None

    def open_on(ticker, date):
        for b in bars.get(ticker, []):
            if b["ts"] == date:
                return b["open"]
        return None

    for date in eval_dates:
        # 1. Fills
        for o in pending:
            px = open_on(o["ticker"], date)
            if px is None:
                continue
            slip = SLIP_LEVERAGED if beta(o["ticker"]) > 1 else SLIP_EQUITY
            if o["side"] == "buy":
                fill = px * (1 + slip)
                qty = o["quantity"]
                cost = fill * qty
                if cost > cash:
                    qty = cash / fill if fill > 0 else 0
                    cost = fill * qty
                if qty <= 0:
                    continue
                held = positions.get(o["ticker"], 0.0)
                prev_cost = avg_cost.get(o["ticker"], 0.0) * held
                positions[o["ticker"]] = held + qty
                avg_cost[o["ticker"]] = (prev_cost + cost) / (held + qty)
                cash -= cost
                trades += 1
            else:
                held = positions.get(o["ticker"], 0.0)
                qty = min(o["quantity"], held)
                if qty <= 0:
                    continue
                fill = px * (1 - slip)
                cash += fill * qty
                positions[o["ticker"]] = held - qty
                trades += 1
        pending = []

        # 2. Mark to Market
        prices = {t: close_on(t, date) for t in bars}
        prices = {t: p for t, p in prices.items() if p is not None}
        pos_value = {t: positions.get(t, 0.0) * prices.get(t, 0.0) for t in positions}
        equity = cash + sum(pos_value.values())
        equity = max(equity, 1e-9)
        equity_curve.append(equity)
        dates.append(date)

        # 3. Call decide
        market_state = {t: [b for b in bars[t] if b["ts"] <= date] for t in bars}
        portfolio_state = {
            "cash": cash,
            "positions": [
                {"ticker": t, "quantity": q, "avg_cost": avg_cost.get(t, 0.0)}
                for t, q in positions.items() if q > 0
            ],
            "last_prices": prices,
        }
        try:
            orders = decide(market_state, portfolio_state, cash)
        except Exception:
            orders = []
        for o in orders or []:
            try:
                if o["side"] in ("buy", "sell") and float(o["quantity"]) > 0 and o["ticker"] in bars:
                    pending.append({"ticker": o["ticker"], "side": o["side"], "quantity": float(o["quantity"])})
            except Exception:
                pass

    # Drawdown curve
    peak = -1e18
    dd_curve = []
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        dd_curve.append(dd * 100) # in percentage

    ret = equity_curve[-1] / START_CASH - 1 if equity_curve else 0.0
    
    # Calculate Max Drawdown
    mdd = 0.0
    peak_val = -1e18
    for v in equity_curve:
        peak_val = max(peak_val, v)
        if peak_val > 0:
            mdd = max(mdd, (peak_val - v) / peak_val)
            
    # Calculate Sharpe
    sharpe = 0.0
    if len(equity_curve) >= 3:
        rets = [equity_curve[j] / equity_curve[j - 1] - 1 for j in range(1, len(equity_curve))]
        mean_ret = sum(rets) / len(rets)
        var_ret = sum((r - mean_ret) ** 2 for r in rets) / (len(rets) - 1)
        sd_ret = math.sqrt(var_ret)
        if sd_ret > 1e-12:
            sharpe = (mean_ret / sd_ret) * math.sqrt(252)

    # Calculate Calmar
    days = len(equity_curve)
    ann_ret = (1 + ret) ** (252 / days) - 1 if days > 0 else 0.0
    calmar = (ann_ret / mdd) if mdd > 1e-9 else 0.0

    return {
        "dates": dates,
        "equity": equity_curve,
        "drawdown": dd_curve,
        "ret": ret * 100,
        "mdd": mdd * 100,
        "sharpe": sharpe,
        "calmar": calmar,
        "trades": trades
    }

def main():
    regimes = json.loads(gzip.open(DATA, "rb").read())
    our_agent = HERE / "agent.py"
    zaid_agent = HERE / "zaid_agent.py"
    
    results = {}
    for name, reg in regimes.items():
        print(f"Running {name}...")
        our_res = run_regime_detailed(our_agent, name, reg)
        zaid_res = run_regime_detailed(zaid_agent, name, reg)
        results[name] = {
            "dates": our_res["dates"],
            "our_equity": our_res["equity"],
            "our_drawdown": our_res["drawdown"],
            "our_stats": {
                "ret": f"{our_res['ret']:.2f}%",
                "mdd": f"{our_res['mdd']:.2f}%",
                "sharpe": f"{our_res['sharpe']:.2f}",
                "calmar": f"{our_res['calmar']:.2f}",
                "trades": our_res["trades"]
            },
            "zaid_equity": zaid_res["equity"],
            "zaid_drawdown": zaid_res["drawdown"],
            "zaid_stats": {
                "ret": f"{zaid_res['ret']:.2f}%",
                "mdd": f"{zaid_res['mdd']:.2f}%",
                "sharpe": f"{zaid_res['sharpe']:.2f}",
                "calmar": f"{zaid_res['calmar']:.2f}",
                "trades": zaid_res["trades"]
            }
        }

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>builderr.ai - AMG Agent Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #0a0e17;
            --card-bg: #131b2e;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --border-color: #1f2937;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            padding: 2rem;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }}
        .logo {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .tabs {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .tab-btn {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .tab-btn:hover {{
            border-color: var(--accent-blue);
            color: var(--text-main);
        }}
        .tab-btn.active {{
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 2rem;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 1.5rem;
        }}
        .stats-sidebar {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        .stats-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }}
        .stat-label {{
            color: var(--text-muted);
        }}
        .stat-value {{
            font-weight: 600;
        }}
        .stat-value.green {{
            color: var(--accent-green);
        }}
        .stat-value.red {{
            color: var(--accent-red);
        }}
        .comparison-badge {{
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            margin-left: 0.5rem;
            font-weight: 600;
        }}
        .comparison-badge.better {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
        }}
        .comparison-badge.worse {{
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--accent-red);
        }}
        .footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1 class="logo">Adaptive Momentum Guard</h1>
                <p style="color: var(--text-muted); margin-top: 0.25rem;">builderr.ai Competition Round 1 Preview</p>
            </div>
            <div class="badge">✓ Admission Safety Bar Cleared</div>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchRegime('calm_uptrend')">Calm Uptrend</button>
            <button class="tab-btn" onclick="switchRegime('moderate_selloff')">Moderate Selloff</button>
            <button class="tab-btn" onclick="switchRegime('vol_spike_snapback')">Vol Spike & Snapback</button>
        </div>

        <div class="grid">
            <div class="card">
                <h2 id="chart-title" style="font-weight: 600;">Equity Curve</h2>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>

            <div class="stats-sidebar">
                <div class="card">
                    <h2 class="stats-title">AMG Agent (Ours)</h2>
                    <div class="stat-row">
                        <span class="stat-label">Return</span>
                        <span id="our-ret" class="stat-value">0.00%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Drawdown</span>
                        <span id="our-mdd" class="stat-value red">0.00%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Calmar Ratio</span>
                        <span id="our-calmar" class="stat-value green">0.00</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Sharpe Ratio</span>
                        <span id="our-sharpe" class="stat-value">0.00</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Trades</span>
                        <span id="our-trades" class="stat-value">0</span>
                    </div>
                </div>

                <div class="card" style="border-color: rgba(59, 130, 246, 0.3);">
                    <h2 class="stats-title" style="color: var(--accent-blue);">Zaid Agent (Benchmark)</h2>
                    <div class="stat-row">
                        <span class="stat-label">Return</span>
                        <span id="zaid-ret" class="stat-value">0.00%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Drawdown</span>
                        <span id="zaid-mdd" class="stat-value red">0.00%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Calmar Ratio</span>
                        <span id="zaid-calmar" class="stat-value">0.00</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Sharpe Ratio</span>
                        <span id="zaid-sharpe" class="stat-value">0.00</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Trades</span>
                        <span id="zaid-trades" class="stat-value">0</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            Adaptive Momentum Guard developed by Gemini AI agent. Double-click dashboard.html to view anytime.
        </div>
    </div>

    <script>
        const data = {json.dumps(results)};
        let activeRegime = 'calm_uptrend';
        let chart = null;

        function initChart(regime) {{
            const ctx = document.getElementById('equityChart').getContext('2d');
            const regimeData = data[regime];
            
            const config = {{
                type: 'line',
                data: {{
                    labels: regimeData.dates,
                    datasets: [
                        {{
                            label: 'AMG (Ours)',
                            data: regimeData.our_equity,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.05)',
                            borderWidth: 3,
                            pointRadius: 0,
                            fill: true,
                            tension: 0.1
                        }},
                        {{
                            label: 'Zaid Agent',
                            data: regimeData.zaid_equity,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.05)',
                            borderWidth: 2,
                            pointRadius: 0,
                            borderDash: [5, 5],
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#f3f4f6',
                                font: {{
                                    family: 'Outfit'
                                }}
                            }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            callbacks: {{
                                label: function(context) {{
                                    return context.dataset.label + ': $' + Math.round(context.raw).toLocaleString();
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{
                                color: '#1f2937'
                            }},
                            ticks: {{
                                color: '#9ca3af',
                                maxTicksLimit: 10,
                                font: {{
                                    family: 'Outfit'
                                }}
                            }}
                        }},
                        y: {{
                            grid: {{
                                color: '#1f2937'
                            }},
                            ticks: {{
                                color: '#9ca3af',
                                font: {{
                                    family: 'Outfit'
                                }},
                                callback: function(value) {{
                                    return '$' + value.toLocaleString();
                                }}
                            }}
                        }}
                    }}
                }}
            }};

            if (chart) {{
                chart.destroy();
            }}
            chart = new Chart(ctx, config);
        }}

        function updateStats(regime) {{
            const stats = data[regime];
            
            document.getElementById('our-ret').textContent = stats.our_stats.ret;
            document.getElementById('our-mdd').textContent = stats.our_stats.mdd;
            document.getElementById('our-calmar').textContent = stats.our_stats.calmar;
            document.getElementById('our-sharpe').textContent = stats.our_stats.sharpe;
            document.getElementById('our-trades').textContent = stats.our_stats.trades;

            document.getElementById('zaid-ret').textContent = stats.zaid_stats.ret;
            document.getElementById('zaid-mdd').textContent = stats.zaid_stats.mdd;
            document.getElementById('zaid-calmar').textContent = stats.zaid_stats.calmar;
            document.getElementById('zaid-sharpe').textContent = stats.zaid_stats.sharpe;
            document.getElementById('zaid-trades').textContent = stats.zaid_stats.trades;
        }}

        function switchRegime(regime) {{
            activeRegime = regime;
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            initChart(regime);
            updateStats(regime);
            document.getElementById('chart-title').textContent = 'Equity Curve - ' + regime.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }}

        // Init
        switchRegime('calm_uptrend');
    </script>
</body>
</html>
"""
    with open(HERE / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Dashboard generated successfully!")

if __name__ == "__main__":
    main()
