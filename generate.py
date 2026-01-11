from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load tokenizer from base GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./model")

model.eval()

prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=80,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))