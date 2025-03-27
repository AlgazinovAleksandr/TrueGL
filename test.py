from transformers import BertForSequenceClassification
model_path = "/Users/mattlaing/Desktop/TruthSeeker/models/finetuned"
model = BertForSequenceClassification.from_pretrained(model_path)
print("Model loaded successfully!")