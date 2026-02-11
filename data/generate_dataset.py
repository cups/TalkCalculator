# Creates training examples.

import json
import random
from .templates import ADD_TEMPLATES
from .number_words import to_words


def make_add_example():
    a, b = random.randint(1, 100), random.randint(1, 100)
    template = random.choice(ADD_TEMPLATES)
    if "{c}" in template:
        c = random.randint(1, 100)
        sentence = template.format(a=to_words(a), b=to_words(b), c=to_words(c))
        numbers = [a, b, c]
    else:
        sentence = template.format(a=to_words(a), b=to_words(b))
        numbers = [a, b]
    # Emit a sequence of tool calls (one add per number)
    tool_calls = [{"name": "add", "arguments": {"number": n}} for n in numbers]
    return {"user": sentence, "tool_calls": tool_calls}


with open("calculator_dataset.jsonl", "w") as f:
    for _ in range(500):
        example = make_add_example()
        f.write(json.dumps(example) + "\n")
