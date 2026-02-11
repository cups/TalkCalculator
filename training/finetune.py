# Skeleton finetune script for FunctionGemma on calculator data

# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

dataset = None  # placeholder - implement loading of ../data/calculator_dataset.jsonl

model_id = "google/functiongemma-270m-it"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

def format_example(example):
    prompt = f"User: {example['user']}\nAssistant: {example['tool_call']}"
    return prompt


# training args and trainer setup to be implemented when data is ready
