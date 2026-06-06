import gzip
import json
import os
import sys
from pathlib import Path
from preview import run_regime, DATA

def main():
    regimes = json.loads(gzip.open(DATA, "rb").read())
    agents = [
        "agent.py",
        "baseline.py",
        "ai_momentum.py",
        "drawdown_momentum.py",
        "example_sector_rotation.py",
        "example_vol_target.py",
        "harsimran_agent.py",
        "mohit_agent.py",
        "momentum_v1.py",
        "opu_agent.py",
        "robert_agent.py",
        "seed_dual_momentum.py",
        "shyam_agent.py",
        "sumegh_agent.py",
        "zaid_agent.py"
    ]
    
    here = Path(__file__).parent
    agents = [a for a in agents if (here / a).exists()]
    
    print(f"=== Benchmarking {len(agents)} agents across 3 regimes ===")
    results_by_agent = {}
    
    for agent_file in sorted(agents):
        agent_path = here / agent_file
        try:
            results = []
            for name, reg in regimes.items():
                res = run_regime(agent_path, name, reg)
                results.append(res)
            results_by_agent[agent_file] = results
        except Exception as e:
            print(f"Failed to run {agent_file}: {e}")
            
    print("\n" + "="*95)
    print(f"{'Agent':<25} | {'Regime':<20} | {'Ret':>8} | {'MaxDD':>8} | {'Calmar':>8} | {'Trades':>6} | {'Err':>3}")
    print("="*95)
    for agent_file, results in sorted(results_by_agent.items()):
        for r in results:
            print(f"{agent_file:<25} | {r['name']:<20} | {r['ret']*100:7.2f}% | {r['mdd']*100:7.2f}% | {r['calmar']:8.2f} | {r['trades']:6d} | {r['errors']:3d}")
        avg_calmar = sum(r['calmar'] for r in results) / len(results)
        worst_dd = max(r['mdd'] for r in results)
        total_ret = sum(r['ret'] for r in results) / len(results)
        print(f"{'-'*25} | {'SUMMARY':<20} | {total_ret*100:7.2f}% | {worst_dd*100:7.2f}% | {avg_calmar:8.2f} | {'':6} | {'':3}")
        print("="*95)

if __name__ == "__main__":
    main()
