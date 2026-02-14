"""Small finetune runner for FunctionGemma on the calculator numeric dataset.

This script is intentionally conservative (small batch, single epoch by default) to
allow a quick local test on CPU or a single GPU. It expects the `datasets` and
`transformers` libraries to be installed.

Example:
  python training/run_small_finetune.py \
      --model_id google/functiongemma-270m-it \
      --dataset ../calculator_dataset.jsonl \
      --output_dir ./training_out \
      --epochs 1

Note: This script does not push any changes or artifacts.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from transformers import BitsAndBytesConfig


def load_jsonl(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_text_examples(examples):
    # Each training example is a single text that contains prompt and target.
    # Prompt: "User: {user}\nAssistant:"  Target: JSON of tool_calls
    texts = []
    for ex in examples:
        user = ex.get('user', '')
        calls = ex.get('tool_calls', [])
        # represent tool_calls compactly as JSON; model will learn to emit this
        target = json.dumps(calls, separators=(',', ':'))
        text = f"User: {user}\nAssistant: {target}\n"
        texts.append({"text": text})
    return texts


def tokenize_function(examples, tokenizer, block_size=512):
    # Tokenize and concatenate prompt+target as single sequence for causal LM
    return tokenizer(examples['text'], truncation=True, max_length=block_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='google/functiongemma-270m-it')
    parser.add_argument('--dataset', default='../calculator_dataset.jsonl')
    parser.add_argument('--output_dir', default='./training_out')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--block_size', type=int, default=512)
    args = parser.parse_args()

    data_path = Path(args.dataset).expanduser()
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    raw = load_jsonl(str(data_path))
    texts = build_text_examples(raw)
    ds = Dataset.from_list(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Ensure tokenizer has pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = ds.map(lambda ex: tokenize_function(ex, tokenizer, args.block_size), batched=True, remove_columns=['text'])
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Load model in 8-bit if possible to save memory
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb, device_map='auto')

    # Prepare model for k-bit training and attach LoRA adapters (PEFT) so fine-tuning is possible
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        print("Attached LoRA adapters for k-bit training")
    except Exception as e:
        print("PEFT not available or failed to configure LoRA adapters:", e)
        print("Install peft and retry: pip install peft")
        raise

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy='no',  # avoid saving for quick local runs unless desired
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training. This may take time depending on hardware.")
    trainer.train()
    print("Training finished. Saving model to output_dir")
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
