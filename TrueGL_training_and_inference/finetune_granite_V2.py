import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# === CONFIG ===
MODEL_PATH = "/root/Fine-Tuning_Truth/granite-3.1-1b-a400m-base" # default
CSV_PATH = "/root/Fine-Tuning_Truth/all_articles_fine_tuning.csv"
OUTPUT_DIR = "/root/Fine-Tuning_Truth/granite-V2-articles"
# bs = 8, and MAX_LENGTH = 1670 works
# bs = 7 and MAX_LENGTH = 1910 works
BATCH_SIZE = 6 # 16 and 8 might be too much, 7 works
EPOCHS = 5
MAX_LENGTH = 2240 # 2240 works
LEARNING_RATE = 3e-5
VAL_SIZE = 0.02   # 2% for validation

# === Custom Dataset ===
class StatementDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = str(self.labels[idx])
        
        # Format the input for causal LM
        # IT IS VERY IMPORTANT TO CHANGE THIS LATER
        full_text = f"Assess the reliability of the following statement on a scale of 0.0 (completely unreliable) to 1.0 (perfectly reliable). Consider factors such as factual accuracy, verifiability, number of alternative viewpoints, logical coherence (the statement is more reliable if: it has no logical contradictions with facts that are easily confirmable, or it has no contradictions within the statement itself), and evidence transparency (if behind a statement there are transparent methods such as statistical data, the statement is more reliable). Provide only the numerical score:\nStatement: {text}\nLabel: {label}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Ensure all tensors are 2D before squeezing
        input_ids = encoding["input_ids"].view(1, -1).squeeze(0)
        attention_mask = encoding["attention_mask"].view(1, -1).squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For causal LM, labels are same as input_ids
        }

def split_data(df, val_size=VAL_SIZE):
    """Split data into train and validation sets, stratified by the labels column."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df['labels'],  # Stratify by the 'labels' column
        random_state=42  # Set a random seed for reproducibility
    )
    return train_df, val_df

def main():
    # === Load Dataset ===
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # === Split Data ===
    print("Splitting dataset...")
    train_df, val_df = split_data(df, VAL_SIZE)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print("Label distribution in train set:")
    print(train_df['labels'].value_counts(normalize=True))

    # === Load Tokenizer ===
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Load Model ===
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    # === Create Datasets ===
    train_dataset = StatementDataset(
        train_df['statement'].tolist(), 
        train_df['labels'].tolist(), 
        tokenizer, 
        MAX_LENGTH
    )
    
    val_dataset = StatementDataset(
        val_df['statement'].tolist(), 
        val_df['labels'].tolist(), 
        tokenizer, 
        MAX_LENGTH
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # === Start Training ===
    print("Starting training...")
    trainer.train()

    # === Evaluate on Val Set ===
    print("Evaluating on val set...")
    val_results = trainer.evaluate(val_dataset)
    print(f"Val results: {val_results}")

    # === Save Model ===
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save test results
    with open(os.path.join(OUTPUT_DIR, "validation_results.txt"), "w") as f:
        f.write(str(val_results))

    print("Training completed successfully!")

if __name__ == "__main__":
    main()