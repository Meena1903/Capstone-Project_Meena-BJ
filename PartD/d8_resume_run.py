"""D8 - Simulate a crash after the second tool, then resume the checkpoint."""
import csv
import json
import os

from d_agent import DATA_DIR, DEMOS, STATE_DIR, SimulatedCrash, log_run, resume_agent, run_agent

with open(os.path.join(DATA_DIR, "raw", "agent_scenarios.csv"), newline="", encoding="utf-8") as source:
    s1 = next(row["message"] for row in csv.DictReader(source) if row["scenario_id"] == "S1")
try:
    run_agent(s1, "S1", run_id="S1-crash-demo", crash_after_tools=2)
except SimulatedCrash as error:
    print(error)
with open(os.path.join(STATE_DIR, "S1-crash-demo.json"), encoding="utf-8") as checkpoint:
    saved = json.load(checkpoint)
before = list(saved["tools_executed"])
result = resume_agent("S1-crash-demo")
after = [step["tool"] for step in result["steps"]][len(before):]
all_tools = before + after
assert all_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert result["status"] == "REFUND" and abs(result["refund_inr"] - 128.25) < 0.01
existing = []
if os.path.exists(DEMOS):
    with open(DEMOS, encoding="utf-8") as source:
        existing = [line for line in source if json.loads(line).get("run_id") != "S1-crash-demo"]
with open(DEMOS, "w", encoding="utf-8") as output:
    output.writelines(existing)
log_run(DEMOS, result)
print("before crash:", before)
print("after resume:", after)
print("all four calls:", all_tools)
print("final:", result["status"], result["refund_inr"])
