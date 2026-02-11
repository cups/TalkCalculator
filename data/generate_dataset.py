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
        number = a
    else:
        sentence = template.format(a=to_words(a), b=to_words(b))
        number = a
    # Emit a single 'number' argument; when templates contain multiple
    # values the dataset currently encodes the first value as the example
    # tool call. Model/agent training can produce multiple add() calls if
    # desired during inference.
    tool_call = {
        "name": "add",
        "arguments": {"number": number}
    }
    return {"user": sentence, "tool_call": tool_call}


with open("calculator_dataset.jsonl", "w") as f:
    for _ in range(500):
        example = make_add_example()
        f.write(json.dumps(example) + "\n")
