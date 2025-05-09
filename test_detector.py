from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ✅ Load model and tokenizer
model_path = "detector/bert_detector_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# 🎣 Take user input
email_text = input("📨 Email: ")

# ✏️ Tokenize
inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)

# 🔍 Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    phishing_confidence = probs[0][1].item()
    legitimate_confidence = probs[0][0].item()

# 🧠 Prediction based on threshold
threshold = 0.6
if phishing_confidence > threshold:
    label = "Phishing"
else:
    label = "Legitimate"

# 📊 Output
print(f"\n🧠 Prediction: {label}")
print(f"🔐 Phishing Confidence: {phishing_confidence:.2f}")
print(f"📬 Legitimate Confidence: {legitimate_confidence:.2f}")
