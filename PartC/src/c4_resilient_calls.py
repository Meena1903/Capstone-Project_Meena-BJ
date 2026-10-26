"""C4 - Demonstrate retries using the full zero-shot run and report recovery counts."""
from common import ZERO_SHOT, run_pipeline

_, stats = run_pipeline("C4-retry-demo", ZERO_SHOT)
print("C4 transient errors:", stats["transient"], "recovered:", stats["recovered"],
      "retry lines:", len(stats["log"]))
