# Creates training examples.

import json
import random
from templates import ADD_TEMPLATES
from number_words import to_words


def make_add_example():
    a, b = random.randint(1, 100), random.randint(1, 100)
    sentence = random.choice(ADD_TEMPLATES).format(a=to_words(a), b=to_words(b))
    tool_call = {
        "name": "add",
        "arguments": {"numbers": [a, b]}
    }
    return {"user": sentence, "tool_call": tool_call}


with open("calculator_dataset.jsonl", "w") as f:
    for _ in range(500):
        example = make_add_example()
        f.write(json.dumps(example) + "\n")
