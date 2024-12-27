from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import nltk
nltk.download('punkt')
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
coref_pipeline = pipeline("coreference-resolution", model=model, tokenizer=tokenizer)
text ="""
Mutiso alikwenda bustanini. Alifurahia matembezi hayo.
# Annotations:
# 1. Maria -> Entity 1
# 2. Alifurahia -> Refers to Entity 1 ("Maria")
# 3. Hayo -> Refers to Entity 1's action ("matembezi")
"""
sentences = nltk.sent_tokenize(text)
print("Tokenized Sentences:")
print(sentences)
resolved_text = coref_pipeline(text)
print("\nResolved Text:")
print(resolved_text)
# comment: Output full context
print("\nFull Processed Output:")
print(resolved_text["resolved"])