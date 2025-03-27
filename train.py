from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd

class LiarDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.encodings = tokenizer(self.data['statement'].tolist(), truncation=True, padding=True)
        self.labels = self.data['label'].tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    train_dataset = LiarDataset('/Users/mattlaing/Desktop/TruthSeeker/data/processed/train.csv', tokenizer)
    valid_dataset = LiarDataset('/Users/mattlaing/Desktop/TruthSeeker/data/processed/valid.csv', tokenizer)

    training_args = TrainingArguments(
        output_dir='./models/finetuned',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    model.save_pretrained('/Users/mattlaing/Desktop/TruthSeeker/models/finetuned')
    tokenizer.save_pretrained('/Users/mattlaing/Desktop/TruthSeeker/models/finetuned')
    print("Model training complete.")

if __name__ == "__main__":
    train_model()