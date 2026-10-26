"""C3 - Validate and report the zero-shot and few-shot prompts."""
from common import ALLOWED, FEW_SHOT, ZERO_SHOT, est_tokens

for prompt in (ZERO_SHOT, FEW_SHOT):
    assert "CLASSIFY" in prompt and all(category in prompt for category in ALLOWED)
    assert '{"category": "...", "reason": "..."}' in prompt
assert "EXAMPLES:" not in ZERO_SHOT
assert "EXAMPLES:" in FEW_SHOT and '"fraud"' in FEW_SHOT
print("C3 prompt tokens: zero-shot", est_tokens(ZERO_SHOT), "few-shot", est_tokens(FEW_SHOT))
