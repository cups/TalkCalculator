# Main loop: user → model → dispatcher → calculator.

import torch
from inference.load_model import load
from inference.prompt_builder import build_prompt
from app.dispatcher import dispatch


tokenizer, model = None, None
try:
    tokenizer, model = load()
except Exception as e:
    print("Warning: model load failed:", e)

if __name__ == '__main__':
    if tokenizer is None:
        print("Model not available, exiting.")
    else:
        while True:
            user_input = input(">> ")
            prompt = build_prompt(user_input)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Model raw:", response)
            try:
                result = dispatch(response)
                if result is not None:
                    print("Calculator:", result)
            except Exception as e:
                print("Dispatch error:", e)
