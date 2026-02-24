"""
=============================================================================
  AI FOR BUSINESS — Week 01
  Topic : Marketer to Machine — Smart Email Compose
  File  : 01.py

  Approach : GPT-2 Language Model for Email Completion
    1. Load pre-trained GPT-2 model and tokenizer (HuggingFace Transformers)
    2. Encode an email prompt into token IDs
    3. Generate the next words / sentences using sampling

  Dependencies : transformers, torch  (pip install transformers torch)
=============================================================================
"""

"""Marketer to Machine: Develop a ML model for Smart email Compose. Smart email compose finishes
sentences for you by predicting what word or words user will type next."""
# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
# Function to complete the email
def complete_email(prompt, max_length=50):
# Encode the input prompt into tokens
input_ids = tokenizer.encode(prompt, return_tensors='pt')
# Generate text with the model, using the input prompt
output = model.generate(input_ids, max_length=max_length,
num_return_sequences=1, do_sample=True)
# Decode the generated text back into readable words
completed_text = tokenizer.decode(output[0], skip_special_tokens=True)
return completed_text
# Example usage
prompt = "Dear Team, I would like to"
completed_email = complete_email(prompt)
print("Completed Email: ", completed_email)
