# Ref: https://pub.towardsai.net/fine-tuning-a-llama-2-7b-model-for-python-code-generation-865453afdf73
# pip3 install "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.1" --upgrade
# pip3 install ipywidgets==7.7.1
# pip3 install huggingface_hub
# pip3 install python-dotenv

from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

from trl import SFTTrainer


# Global parameter setting
model_id = "NousResearch/Llama-2-7b-hf"         # The model that you want to train from the Hugging Face hub
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"   # The instruction dataset to use
#dataset_name = "HuggingFaceH4/CodeAlpaca_20K"

dataset_split= "train"                          # Dataset split
new_model = "llama-2-7b-int4-python-code-20k"   # Fine-tuned model name
hf_model_repo="edumunozsala/"+new_model         # Huggingface repository
device_map = {"": 0}                            # Load the entire model on the GPU 0

################################################################################
# bitsandbytes parameters
################################################################################
use_4bit = True                         # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"      # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"             # Quantization type (fp4 or nf4)
use_double_nested_quant = False         # Activate nested quantization for 4-bit base models (double quantization)

################################################################################
# QLoRA parameters
################################################################################
lora_r = 64                             # LoRA attention dimension
lora_alpha = 16                         # Alpha parameter for LoRA scaling
lora_dropout = 0.1                      # Dropout probability for LoRA layers

################################################################################
# TrainingArguments parameters
################################################################################
output_dir = new_model              # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 1                # Number of training epochs
fp16 = False                        # Enable fp16/bf16 training (set bf16 to True with an A100)
bf16 = True
per_device_train_batch_size = 4     # Batch size per GPU for training
gradient_accumulation_steps = 1 # 2 # Number of update steps to accumulate the gradients for
gradient_checkpointing = True       # Enable gradient checkpointing
max_grad_norm = 0.3                 # Maximum gradient normal (gradient clipping)
learning_rate = 2e-4 #1e-5          # Initial learning rate (AdamW optimizer)
weight_decay = 0.001                # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"         # Optimizer to use
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant"
max_steps = -1                      # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03                 # Ratio of steps for a linear warmup (from 0 to learning rate)
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False
save_steps = 0                      # Save checkpoint every X updates steps
logging_steps = 25                  # Log every X updates steps
disable_tqdm= True                  # Disable tqdm

################################################################################
# SFTTrainer parameters
################################################################################
max_seq_length = 2048 #None         # Maximum sequence length to use
packing = True #False               # Pack multiple short examples in the same input sequence to increase efficiency

token = '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'

# Load dataset from the hub
dataset = load_dataset(dataset_name, split=dataset_split)
# Show dataset size
print(f"dataset size: {len(dataset)}")
# Show an example
print(dataset[randrange(len(dataset))])

