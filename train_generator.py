import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ============================================================
# 1. LOAD CLEANED + BALANCED EMAILS
# ============================================================
csv_path = "data/processed/balanced_cleaned_emails.csv"
df = pd.read_csv(csv_path)

# TRAIN ONLY ON PHISHING EMAILS
phishing_emails = df[df["label"] == 1]["text"].tolist()

print(f"Loaded {len(phishing_emails)} phishing emails for training.")

# ============================================================
# 2. LOAD TOKENIZER & MODEL
# ============================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# ============================================================
# 3. DATASET CLASS
# ============================================================
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
        )

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encodings["input_ids"][idx])
        attention_mask = torch.tensor(self.encodings["attention_mask"][idx])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = EmailDataset(phishing_emails, tokenizer)

# ============================================================
# 4. TRAINING SETTINGS
# ============================================================
training_args = TrainingArguments(
    output_dir="generator_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    warmup_steps=100,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-style training
)

# ============================================================
# 5. TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ============================================================
# 6. TRAIN & SAVE MODEL
# ============================================================
trainer.train()

model.save_pretrained("generator_model")
tokenizer.save_pretrained("generator_model")

print("\n✅ GPT-2 phishing email generator trained and saved successfully!")
