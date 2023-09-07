
from transformers import AutoTokenizer, AutoModelForMaskedLM


import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("imvladikon/alephbertgimmel-base-512")
tokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")

text = "{} היא מטרופולין המהווה את מרכז הכלכלה"

input = tokenizer.encode(text.format("[MASK]"), return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(text.format(tokenizer.decode([token])))
