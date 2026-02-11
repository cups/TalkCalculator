# an example showing how 8 bit functionGemma cannot seeem to 
# understand simple arithmetic on its own


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/functiongemma-270m-it"

# ---- Quantization config ----
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model in 8-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()
print("Model loaded.\n")


prompt = "Q:  Add 2 + 2:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""
# ---- Simple generation test ----
prompt = "Hello! What is 2 plus 2?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7
    )

print("=== Model Output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

