# Main loop: user → model → dispatcher → calculator.

import json
import torch
from inference.load_model import load
from inference.prompt_builder import build_prompt
from app.dispatcher import dispatch


tokenizer, model = None, None
try:
    tokenizer, model = load()
except Exception as e:
    print("Warning: model load failed:", e)


def _extract_json(text: str) -> str:
    """Extract the first top-level JSON object or array from the model output.

    This is a lightweight heuristic: find the first '{' or '[' and the last
    matching '}' or ']' and return the substring. If not found, return the
    original text.
    """
    s = text.strip()
    start = None
    open_char = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            open_char = ch
            break
    if start is None:
        return s
    close_char = '}' if open_char == '{' else ']'
    end = s.rfind(close_char)
    if end == -1:
        return s
    return s[start:end+1]


if __name__ == '__main__':
    if tokenizer is None:
        print("Model not available, exiting.")
    else:
        while True:
            user_input = input(">> ")
            prompt = build_prompt(user_input)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Model raw:", response)
            json_part = _extract_json(response)
            try:
                # dispatch expects a JSON string
                result = dispatch(json_part)
                if result is not None:
                    print("Calculator:", result)
            except Exception as e:
                print("Dispatch error:", e)
