from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-de"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Input text in English
text = "Translate this text to German."

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Translate text to German
outputs = model.generate(**inputs)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("English Text:", text)
print("Translated Text (German):", translated_text)
