# Linking functionGemma to Calculator

# PLAN

We have a functioning and robust python Calculator class 
(https://github.com/cups/Calculator/tree/main)
Now we want to insert functionGemma between a simple interface 
which accepts initially numbers but eventually typed strings, as if they were spoken text
and send them to the Calculator class.
To do that we need to fine tune functionGemma by generating 
a corpus of training data based on strings

    e.g. in Phase 1
    "What is 7 and 9?"
    (functionGemma calls add(7) and add(9)
    calc.add(7)
    calc.add(9)
    calc.get_total() # shows int 16 (i.e NOT "sixteen")

    e.g. in Phase 2
    "What is seven and nine?"
    (functionGemma calls add(7) and add(9)
    calc.add(7)
    calc.add(9)
    calc.get_total() # shows int 16 (i.e NOT "sixteen")

## Tasks
This folder is a python venv, all functionGemma files are contained - done
functionGemma is currently working - done  
TODO 
create folders and documents as outlined at end of this file
rm the now defunct file suggestions at the end of this file when that job is checked and done 
create git repo
create README file
create .gitignore



### Phase 1  Does the concept work?

Create an intial simple example with just add() get_total() and clear_all() methods
Store synthetic data in an sqlite db in /data folder
 - Start of with pairs of digits 
 - Create a corpus of test data, mostly centred on the add() method
 - Add a column to the db which contains the correct answer for testing, "add_results" outcomes e.g. 16
interpolate the numbers into the templates in the first instance that would be
"What is 7 and 9?" using the previous example
Create training data as JSON
Fine tune functionGemma to recognise the add() tasks 
Add a nice single page GUI to interact with the data and create edge test cases 

### Phase 2 Does the concept work with string numbers?

create synthetic data i.e. text strings as instructions e.g. "Add seven and nine"  
Add more columns to the db and turn the test data into strings
Then add the ability to extract the numbers from the text in python

### Phase 3 Does the concept work with complex number strings?

Add ability to extract complex numbers from text e.g. decimals, thousands and hundreds etc

### Phase 4 Add the methods subtract(), multiply() and divide()

Gradually add other methods also using TDD 

#### Tools and methods
python 
pytest   
linter  
functionGemma quantized to 8 bit
HF tensors  

### File layout (where functionGemma/ is this folder


	functionGemma/
	│
	├── models/
	│   └── (HF cache lives here automatically)
	│
	├── data/
	│   ├── templates.py
	│   ├── number_words.py
	│   ├── generate_dataset.py
	│   └── calculator_dataset.jsonl
	│
	├── training/
	│   └── finetune.py
	│
	├── inference/
	├── load_model.py
	│   ├── prompt_builder.py
	│   └── run_agent.py
	│
	├── app/
	│   ├── dispatcher.py
	│   └── calculator_interface.py
	│
	├── tests/
	│
	└── requirements.txt
	│
	└── fg_tests.py 
	│
	└── load_functionGemma_8bit.py
	│
	└── copilot-instructions.md 




Below are eExamples of file content mostly missing, then add the suggested content as python comments to the files, again for those that are missing from this directory tree. 



### file : requirements.txt contains e.g.

torch
transformers
accelerate
bitsandbytes
datasets
num2words


### file : fg_tests.py contains e.g.

an example of FG usage where it fails to do basic
arithmetic on its own

### file : load_functionGemma_8bit.py contains e.g.

a simple file which proves the correct model is loaded

### file : app/dispatcher.py contains e.g.

# Takes model output JSON → calls calculator methods.

import json
from calculator_interface import call_tool

def dispatch(model_output: str):
    """
    Expects: {"name":"add","arguments":{"number":27}}
    """
    tool_call = json.loads(model_output)
    result = call_tool(tool_call["name"], tool_call.get("arguments", {}))
    return result


### file : app/calculator_interface.py contains e.g.

#Wraps the  existing calculator so the rest of the system doesn’t care about internals.

from calculator import Calculator  # your existing class

calculator = Calculator()

def call_tool(name, args):
    if name == "add":
        calculator.add(args["number"])
    elif name == "get_total":
        return calculator.get_total()
    elif name == "clear_all":
        calculator.clear_all()
    else:
        raise ValueError(f"Unknown tool: {name}")

### file : inference/load_model.py contains e.g.

# Loads FunctionGemma in 8-bit.

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/functiongemma-270m-it"

def load():
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

### file : inference/prompt_builder.py contains e.g.

# Builds structured prompts for function calling.

SYSTEM_PROMPT = """You are a system that converts calculator requests into JSON function calls.

Available functions:

add(number()
get_total()
clear_all()

Respond ONLY with JSON.
"""

def build_prompt(user_input: str):
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"

### file : inference/run_agent.py contains e.g.

# Main loop: user → model → dispatcher → calculator.

import torch
from load_model import load
from prompt_builder import build_prompt
from app.dispatcher import dispatch

tokenizer, model = load()

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

### file :  data/templates.py contains e.g.

# Sentence templates for synthetic data.

ADD_TEMPLATES = [
    "What is {a} and {b}?",
    "How much is {a} and {b}?",
    "Add {a} and {b} makes?",
    "Add {a}, {b} and {c}, show total",
    "Add {a} plus {b} plus  {c} what does that come to?",
]

TOTAL_TEMPLATES = [
    "What is the total?",
    "Current total?",
]

CLEAR_TEMPLATES = [
    "Clear everything",
    "Reset calculator"
]

### file : 🔢 data/number_words.py contains e.g.

# Converts numbers to words to create synthetic data

from num2words import num2words

def to_words(n):
    return num2words(n)

### file : data/generate_dataset.py contains e.g.

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

### file : training/finetune.py (skeleton) contains e.g.

# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

dataset = load_dataset("json", data_files="../data/calculator_dataset.jsonl")

model_id = "google/functiongemma-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def format_example(example):
    prompt = f"User: {example['user']}\nAssistant: {example['tool_call']}"
    return tokenizer(prompt, truncation=True)

dataset = dataset.map(format_example)

training_args = TrainingArguments(
    output_dir="./fg-calculator",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
trainer.train()

