"""C5 - Demonstrate the JSON, keyword and fail-closed output validation rules."""
from common import validate

examples = ['{"category":"billing","reason":"charge"}',
            "The customer has a billing-related problem.",
            '{"category":"shipping","reason":"wrong enum"}', "no category here"]
for example in examples:
    print(example, "->", validate(example))
