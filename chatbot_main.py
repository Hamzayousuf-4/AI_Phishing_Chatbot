import random
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification

# Load GPT-2 phishing email generator
gpt2_path = "generator/final_gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model.eval()

# Load BERT phishing detector
bert_path = "detector/bert_detector_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
bert_model = BertForSequenceClassification.from_pretrained(bert_path, local_files_only=True)
bert_model.eval()

# Predefined prompts and fake senders/subjects
random_prompts = [
    "Reset your credentials urgently",
    "Your package delivery failed",
    "Security alert: unusual login detected",
    "Verify your email account now",
    "You've won a reward! Claim now"
]

fake_senders = [
    "Apple Support", "Google Security", "PayPal Help Center", "Bank of America", "Amazon Support"
]

fake_subjects = [
    "Immediate Action Required", "Suspicious Login Attempt", "Your Account Has Been Locked",
    "Payment Failed", "Claim Your Prize Now"
]

def clean_output(text):
    # Removes only CNN links from the generated text
    return re.sub(r"http[s]?://(?:www\.)?cnn\.com\S*", "[removed CNN link]", text)

def generate_email(prompt):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2_model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    return clean_output(gpt2_tokenizer.decode(output[0], skip_special_tokens=True))

def classify_email(email_text):
    inputs = bert_tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        phishing_conf = probs[0][1].item()
        legit_conf = probs[0][0].item()
    label = "Phishing" if phishing_conf > 0.6 else "Legitimate"
    return label, phishing_conf, legit_conf

def main():
    while True:
        print("\n💬 Menu:")
        print("1. Generate phishing email (user prompt)")
        print("2. Generate phishing email (random prompt)")
        print("3. Detect if an email is phishing or legitimate")
        print("0. Exit")

        choice = input("\n🔢 Enter your choice: ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        elif choice == "1":
            print("\n📝 Custom Phishing Email Setup")
            sender = input("📧 Enter fake sender name (e.g. PayPal Support): ").strip()
            subject = input("📝 Enter email subject (e.g. Urgent Account Update): ").strip()
            message_type = input("📂 Email type or concern (e.g. password reset, invoice): ").strip()
            prompt = f"From: {sender}\nSubject: {subject}\n{message_type}"
            generated = generate_email(prompt)
            print("\n📨 Generated Email:\n" + generated)

        elif choice == "2":
            prompt = random.choice(random_prompts)
            sender = random.choice(fake_senders)
            subject = random.choice(fake_subjects)
            print(f"\n🎲 Random Prompt: {prompt}")
            full_prompt = f"From: {sender}\nSubject: {subject}\n{prompt}"
            generated = generate_email(full_prompt)
            print("\n📨 Generated Email:\n" + generated)

        elif choice == "3":
            user_input = input("\n✉️ Enter email text to classify: ").strip()
            label, phishing_conf, legit_conf = classify_email(user_input)
            print(f"\n🧠 Prediction: {label}")
            print(f"🔐 Phishing Confidence: {phishing_conf:.2f}")
            print(f"📩 Legitimate Confidence: {legit_conf:.2f}")

        else:
            print("❌ Invalid choice. Please select 0, 1, 2, or 3.")

if __name__ == "__main__":
    main()
