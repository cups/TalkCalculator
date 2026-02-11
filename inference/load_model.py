# Loads FunctionGemma in 8-bit.

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

MODEL_ID = "google/functiongemma-270m-it"


def load():
    bnb_config = None
    if BitsAndBytesConfig is not None:
        try:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model
