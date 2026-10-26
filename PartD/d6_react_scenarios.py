"""D6 - Run the seven refund scenarios through the ReAct agent."""
import csv
import json
import os

from d_agent import DATA_DIR, DEMOS, LEDGER, RUNS, STATE_DIR, log_run, run_agent

run_ids = [f"S{number}-main" for number in range(1, 8)] + [
    "S1-token-limit-demo", "S1-timeout-demo", "S1-crash-demo", "S7-stop-demo"]
for path in (LEDGER, RUNS, DEMOS, *(os.path.join(STATE_DIR, f"{run_id}.json") for run_id in run_ids)):
    if os.path.exists(path):
        os.remove(path)
with open(os.path.join(DATA_DIR, "raw", "agent_scenarios.csv"), newline="", encoding="utf-8") as source:
    scenarios = list(csv.DictReader(source))
with open(RUNS, "w", encoding="utf-8"):
    pass
for scenario in scenarios:
    result = run_agent(scenario["message"], scenario["scenario_id"], run_id=f"{scenario['scenario_id']}-main")
    log_run(RUNS, result)
    expected = scenario["expected_outcome"]
    status_ok = result["status"].startswith("FALLBACK") if expected == "FALLBACK" else result["status"] == expected
    amount_ok = abs(result["refund_inr"] - float(scenario["expected_refund_inr"])) <= 0.01
    print(scenario["scenario_id"], result["status"], result["refund_inr"],
          result["llm_calls"], result["tokens_used"], "PASS" if status_ok and amount_ok else "FAIL")
print("saved outputs/agent_runs.jsonl")
