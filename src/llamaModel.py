from unsloth import FastLanguageModel # FastLanguageModel for LLMs
from transformers import TextStreamer

import torch

max_seq_length = 12048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Llama-3.1-8B-bnb-4bit",     
#     "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
#     "unsloth/Llama-3.1-70B-bnb-4bit",
#     "unsloth/Llama-3.1-405B-bnb-4bit",     
# ] 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    max_seq_length = max_seq_length, # Choose any! We auto support RoPE Scaling internally!
    device_map="cuda"
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

FastLanguageModel.for_inference(model) # Enable for inference!

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


inputs = tokenizer(
[
    prompt.format(
        "Answer like a human would in an engaging way", # instruction
        "Hello, how are you", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

