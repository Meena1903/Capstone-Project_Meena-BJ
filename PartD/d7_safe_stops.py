"""D7 - Demonstrate max-iteration, token-limit and timeout fallbacks."""
import csv
import json
import os

from d_agent import DATA_DIR, DEMOS, log_run, run_agent

with open(os.path.join(DATA_DIR, "raw", "agent_scenarios.csv"), newline="", encoding="utf-8") as source:
    scenarios = {row["scenario_id"]: row["message"] for row in csv.DictReader(source)}
s1 = scenarios["S1"]
looping = run_agent(scenarios["S7"], "S7", run_id="S7-main")
token = run_agent(s1, "S1", token_limit=200, run_id="S1-token-limit-demo")
timeout = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
with open(DEMOS, "w", encoding="utf-8") as output:
    for result in (token, timeout):
        output.write(json.dumps({"scenario_id": result["scenario_id"], "run_id": result["run_id"],
                                 "status": result["status"], "final_answer": result["final_answer"],
                                 "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                                 "steps": result["steps"]}) + "\n")
print("S7:", looping["status"], looping["llm_calls"], "calls", looping["tokens_used"], "tokens")
print("S1 token limit:", token["status"], token["llm_calls"], "calls", token["tokens_used"], "tokens")
print("S1 timeout:", timeout["status"], timeout["llm_calls"], "calls")
