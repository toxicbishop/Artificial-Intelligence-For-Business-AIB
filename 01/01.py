# Import necessary libraries 
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
import torch 
# Load pre-trained GPT-2 model and tokenizer 
model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name) 
model = GPT2LMHeadModel.from_pretrained(model_name) 
# Function to complete the email 
def complete_email(prompt, max_length=50):  # Function to complete the email
	input_ids = tokenizer.encode(prompt, return_tensors='pt') # Encode the input prompt into tokens 
	output = model.generate(input_ids, max_length=max_length,num_return_sequences=1, do_sample=True) # Generate text with the model, using the input prompt
	completed_text = tokenizer.decode(output[0], skip_special_tokens=True) # Decode the generated text back into readable words
	return completed_text 
# Example usage 
prompt = "Dear Team, I would like to" 
completed_email = complete_email(prompt, max_length=100) 
print("Completed Email: ", completed_email)

#Output
#Completed Email:  Dear Team, I would like to express congratulations on your success in the last few weeks here at BGG - your #feedback has been amazing and I thank all of our community members for their time and commitment to our work. As always we #respect your continued