import pandas as pd
import os

# Paths
phishing_path = 'data/raw/small_phishing_emails.csv'
legit_path = 'data/raw/small_legitimate_emails.csv'
output_path = 'data/processed/cleaned_emails.csv'

# Load datasets
phishing_df = pd.read_csv(phishing_path)
legit_df = pd.read_csv(legit_path)

# Combine both and keep only useful columns
combined_df = pd.concat([phishing_df, legit_df], ignore_index=True)
combined_df = combined_df[['subject', 'body', 'label']]

# Fill missing text
combined_df = combined_df.fillna('')

# Combine subject and body
combined_df['text'] = combined_df['subject'] + ' ' + combined_df['body']

# Basic cleaning
def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

combined_df['text'] = combined_df['text'].apply(clean_text)

# Save cleaned dataset
processed_df = combined_df[['text', 'label']]
os.makedirs('data/processed', exist_ok=True)
processed_df.to_csv(output_path, index=False)

print(f"✅ Cleaned data saved to {output_path}")
# Save phishing text for GPT-2 (short, clean lines)
phishing_only = processed_df[processed_df['label'] == 1]

with open('data/processed/phishing_train.txt', 'w', encoding='utf-8') as f:
    for line in phishing_only['text']:
        cleaned = line.strip().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
        # Keep only reasonably short entries
        if 20 < len(cleaned) < 300:
            f.write(cleaned + '\n')

print("✅ Filtered phishing samples saved to phishing_train.txt")
