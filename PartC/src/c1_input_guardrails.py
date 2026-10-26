"""C1 - Report the empty and overlong ticket guardrails."""
from common import MAX_INPUT_TOKENS, inputs, est_tokens

tickets, _ = inputs()
empty = tickets.text.str.strip().eq("")
long = tickets.text.map(est_tokens).gt(MAX_INPUT_TOKENS)
print("C1 tickets:", len(tickets))
print("skipped_empty:", int(empty.sum()), "| model calls: 0")
print("flagged_long:", int(long.sum()), "| max tokens:", int(tickets.loc[long, "text"].map(est_tokens).max()), "| model calls: 0")
