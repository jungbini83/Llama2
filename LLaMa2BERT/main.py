# https://towardsdatascience.com/topic-modeling-with-llama-2-85177d01e174
# pip install bertopic datasets accelerate bitsandbytes xformers adjustText

from datasets import load_dataset
from torch import cuda
from torch import bfloat16
import transformers

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
model_id = 'meta-llama/Llama-2-13b-chat-hf'
ACCESS_TOKEN = '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'

# Quantization to load an LLM with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit quantization
    bnb_4bit_quant_type='nf4',          # Normalized float 4
    bnb_4bit_use_double_quant=True,     # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16,     # Computation type
)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
    token=ACCESS_TOKEN,
)
model.eval()