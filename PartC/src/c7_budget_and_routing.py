"""C7 - Run the token-budget demo and final few-shot routing output."""
from common import FEW_SHOT, OUTPUTS, TOKEN_BUDGET, ZERO_SHOT, run_pipeline, summarize, write_retry_lines

budget, budget_stats = run_pipeline("budget-demo", ZERO_SHOT, token_budget=TOKEN_BUDGET)
summarize(f"Budget demo, limit {TOKEN_BUDGET}", budget, budget_stats)
exhausted = int((budget.status == "budget_exhausted").sum())
assert exhausted > 0
final, final_stats = run_pipeline("final", FEW_SHOT, token_budget=100000)
summarize("Final few-shot", final, final_stats)
columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
budget[columns].to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
final[columns].to_csv(OUTPUTS / "triage_results.csv", index=False)
print("budget-exhausted tickets:", exhausted)
print("final routing:\n", final.groupby(["route", "priority"]).size())
write_retry_lines(OUTPUTS / "retry_log.txt", [budget_stats, final_stats], mode="w")
retry_path = OUTPUTS / ".triage_retry_part.log"
if retry_path.exists():
    with retry_path.open(encoding="utf-8") as prior, (OUTPUTS / "retry_log.txt").open("a", encoding="utf-8") as output:
        output.writelines(prior.readlines())
    retry_path.unlink()
print("saved outputs/triage_budget_demo.csv, triage_results.csv and retry_log.txt")
