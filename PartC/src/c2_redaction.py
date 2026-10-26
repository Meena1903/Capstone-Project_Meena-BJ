"""C2 - Redact PII in the required order and save an inspectable intermediate."""
from common import OUTPUTS, PATTERNS, inputs, redact

tickets, _ = inputs()
counts = {name: 0 for name, _, _ in PATTERNS}
cleaned = []
for row in tickets.itertuples():
    text, found = redact(row.text)
    for name, count in found.items():
        counts[name] += count
    cleaned.append(text)
left = sum(len(pattern.findall(text)) for text in cleaned for _, pattern, _ in PATTERNS)
assert left == 0
tickets["text"] = cleaned
tickets.to_csv(OUTPUTS / "redacted_tickets.csv", index=False)
print("C2 substitutions:", counts, "total:", sum(counts.values()), "remaining matches:", left)
