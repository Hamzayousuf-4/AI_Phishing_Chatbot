import pandas as pd
import os

# Paths
input_file = "data/raw/CEAS_08.csv"
output_file = "data/processed/balanced_cleaned_emails.csv"

# Load CSV
df = pd.read_csv(input_file)

# Fill NaN and combine 'subject' and 'body' into 'text'
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)

# Filter to keep only rows with label 0 or 1
df = df[df['label'].isin([0, 1])]

# Drop rows with known noisy/log content
noise_keywords = [
    "clamav", "virustotal", "submission-id", "author:", "svn commit", "log message:",
    "trojan.spy", "virus name alias", "added: no", "spamassassin", "svn", ".py", ".cvd"
]
pattern = '|'.join(noise_keywords)
df = df[~df['text'].str.lower().str.contains(pattern, na=False)]

# Drop empty or whitespace-only emails
df = df[df['text'].str.strip().astype(bool)]

# Drop duplicates
df = df.drop_duplicates(subset=['text'])

# Print cleaned label distribution
print("\n📊 Label value counts after cleaning:")
print(df['label'].value_counts())

# Balance dataset to 9182 phishing + 9182 legitimate
phishing_df = df[df['label'] == 1].sample(n=9182, random_state=42)
legit_df = df[df['label'] == 0].sample(n=9182, random_state=42)
balanced_df = pd.concat([phishing_df, legit_df]).sample(frac=1, random_state=42)

# Save to CSV
os.makedirs("data/processed", exist_ok=True)
balanced_df[['text', 'label']].to_csv(output_file, index=False)

print(f"\n✅ Cleaned and balanced dataset saved to: {output_file}")
