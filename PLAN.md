# Linking functionGemma to Calculator

# PLAN

Note: calculator_dataset.jsonl is currently listed in .gitignore to keep generated datasets local for testing; remove this line from .gitignore and add the dataset to the repo once it has been validated.

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
create folders and documents as outlined at end of this file - done
rm the now defunct file suggestions at the end of this file when that job is checked - done 
create git repo - done
create README file - done
create .gitignore - done



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

    e.g. in Phase 3
    "What is 1,234.56 and 7.89?"
    (functionGemma calls add(1234.56) and add(7.89)
    calc.add(1234.56)
    calc.add(7.89)
    calc.get_total() # shows float 1242.45 (i.e NOT "one thousand two hundred forty-two point four five")

Add ability to extract complex numbers from text e.g. decimals, thousands and hundreds etc

### Phase 4 Add the methods subtract(), multiply() and divide()

    e.g. in Phase 4
    "What is 10 minus 3?"
    (functionGemma calls add(10) and subtract(3)
    calc.add(10)
    calc.subtract(3)
    calc.get_total() # shows int 7

    "Multiply 6 and 7"
    (functionGemma calls multiply(6) and multiply(7)
    calc.multiply(6)
    calc.multiply(7)
    calc.get_total() # shows int 42

    "What is 10 divided by 2?"
    (functionGemma calls add(10) and divide(2)
    calc.add(10)
    calc.divide(2)
    calc.get_total() # shows float 5.0

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
	└── fg_tester.py 
	│
	└── load_functionGemma_8bit.py
	│
	└── copilot-instructions.md 


