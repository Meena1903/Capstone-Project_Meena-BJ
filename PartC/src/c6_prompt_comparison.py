"""C6 - Run zero-shot and few-shot classification and compare accuracy and token cost."""
import pandas as pd

from common import FEW_SHOT, OUTPUTS, ZERO_SHOT, run_pipeline, summarize, write_retry_lines

zero, zero_stats = run_pipeline("zero-shot", ZERO_SHOT)
few, few_stats = run_pipeline("few-shot", FEW_SHOT)
zero_accepted, _ = summarize("Zero-shot", zero, zero_stats)
summarize("Few-shot", few, few_stats)
print("Zero-shot confusion table:")
print(pd.crosstab(zero_accepted.true_category, zero_accepted.category))
print("Few-shot adds tokens per classified ticket; it is worth the cost when avoiding misroutes is more valuable than the added inference cost.")
write_retry_lines(OUTPUTS / ".triage_retry_part.log", [zero_stats, few_stats])
