from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# replace with the model you're using
model_name = 'sshleifer/distilbart-cnn-12-6'

# Download model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model and tokenizer
model.save_pretrained("./" + model_name)
tokenizer.save_pretrained("./" + model_name)
