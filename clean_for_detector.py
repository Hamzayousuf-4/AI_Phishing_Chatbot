import pandas as pd
import os

# Paths
input_file = "data/raw/CEAS_08.csv"
output_file = "data/processed/balanced_cleaned_emails.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Combine subject and body
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)

# Keep only label 0 and 1
df = df[df['label'].isin([0, 1])]

# Remove noise patterns
noise_keywords = [
    "clamav", "virustotal", "submission-id", "author:", "svn commit", "log message:",
    "trojan.spy", "virus name alias", "added: no", "spamassassin", "svn", ".py", ".cvd"
]
pattern = '|'.join(noise_keywords)
df = df[~df['text'].str.lower().str.contains(pattern, na=False)]

# Drop empty or duplicate emails
df = df[df['text'].str.strip().astype(bool)]
df = df.drop_duplicates()

# Count after cleaning
print("\n📊 Label value counts after cleaning:")
print(df['label'].value_counts())

# Balance: get all 0-label and match count of 1-label
legit_df = df[df['label'] == 0]
phishing_df = df[df['label'] == 1].sample(n=len(legit_df), random_state=42)
balanced_df = pd.concat([phishing_df, legit_df]).sample(frac=1, random_state=42)

# Save output
os.makedirs("data/processed", exist_ok=True)
balanced_df[['text', 'label']].to_csv(output_file, index=False)

print(f"\n✅ Balanced dataset saved to {output_file} with {len(balanced_df)} rows.")
