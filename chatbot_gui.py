import random
import re
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification

# Load GPT-2 and BERT models
gpt2_path = "generator/final_gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model.eval()

bert_path = "detector/bert_detector_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
bert_model = BertForSequenceClassification.from_pretrained(bert_path, local_files_only=True)
bert_model.eval()

# Prompts
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

# App initialization
app = ttk.Window(title="📧 AI Phishing Email Tool", themename="flatly", size=(900, 700))
app.resizable(False, False)

# Container frames
frame_landing = ttk.Frame(app)
frame_main = ttk.Frame(app)

frame_landing.pack(fill=BOTH, expand=True)

# Landing Page
ttk.Label(
    frame_landing, 
    text="AI PHISHING EMAIL DETECTION AND GENERATION", 
    font=("Helvetica", 18, "bold")
).pack(pady=(60, 10))

ttk.Label(
    frame_landing,
    text="BY Hamza Yousuf and Sohaib Jaber",
    font=("Helvetica", 14, "italic")
).pack(pady=(0, 40))

ttk.Button(
    frame_landing, 
    text="🚀 Start Chat", 
    bootstyle=SUCCESS, 
    width=20,
    command=lambda: (frame_landing.pack_forget(), frame_main.pack(fill=BOTH, expand=True))
).pack(pady=20)

# Theme toggle
def toggle_theme():
    current = app.style.theme.name
    new_theme = "darkly" if current != "darkly" else "flatly"
    app.style.theme_use(new_theme)

ttk.Button(
    frame_landing, 
    text="🌓 Toggle Dark Mode", 
    bootstyle=SECONDARY, 
    command=toggle_theme
).pack(pady=10)

# Main Interface with Tabs
notebook = ttk.Notebook(frame_main)
notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

# --- TAB 1: Email Generator ---
frame_gen = ttk.Frame(notebook)
notebook.add(frame_gen, text="📨 Generate Email")

ttk.Label(frame_gen, text="Custom Email Generator", font=("Helvetica", 14, "bold")).pack(pady=10)

gen_inputs = ttk.Frame(frame_gen)
gen_inputs.pack(pady=5)

ttk.Label(gen_inputs, text="Sender:").grid(row=0, column=0, sticky=E)
entry_sender = ttk.Entry(gen_inputs, width=50)
entry_sender.grid(row=0, column=1, padx=5, pady=3)

ttk.Label(gen_inputs, text="Subject:").grid(row=1, column=0, sticky=E)
entry_subject = ttk.Entry(gen_inputs, width=50)
entry_subject.grid(row=1, column=1, padx=5, pady=3)

ttk.Label(gen_inputs, text="Email Type / Concern:").grid(row=2, column=0, sticky=E)
entry_concern = ttk.Entry(gen_inputs, width=50)
entry_concern.grid(row=2, column=1, padx=5, pady=3)

# ✅ FIXED: Use user input directly, not GPT
def handle_custom_gen():
    sender = entry_sender.get()
    subject = entry_subject.get()
    concern = entry_concern.get()
    
    if not sender or not subject or not concern:
        messagebox.showwarning("Missing Fields", "Please fill in all fields.")
        return

    result = f"From: {sender}\nSubject: {subject}\n{concern}"
    output_gen.delete("1.0", "end")
    output_gen.insert("end", result)

ttk.Button(gen_inputs, text="🛠 Generate Custom Email", bootstyle=PRIMARY, command=handle_custom_gen).grid(row=3, column=1, pady=10)

ttk.Button(frame_gen, text="🎲 Generate Random Email", bootstyle=INFO, command=lambda: (
    output_gen.delete("1.0", "end"),
    output_gen.insert("end", generate_email(f"From: {random.choice(fake_senders)}\nSubject: {random.choice(fake_subjects)}\n{random.choice(random_prompts)}"))
)).pack(pady=5)

output_gen = ttk.Text(frame_gen, height=12, width=100)
output_gen.pack(padx=10, pady=10)

# === NEW BUTTONS ADDED HERE ===
gen_buttons = ttk.Frame(frame_gen)
gen_buttons.pack(pady=5)

def copy_to_clipboard():
    content = output_gen.get("1.0", "end").strip()
    app.clipboard_clear()
    app.clipboard_append(content)
    messagebox.showinfo("Copied", "Email content copied to clipboard!")

def import_to_detector():
    content = output_gen.get("1.0", "end").strip()
    input_detect.delete("1.0", "end")
    input_detect.insert("end", content)
    notebook.select(frame_detect)

ttk.Button(gen_buttons, text="📋 Copy to Clipboard", bootstyle=SECONDARY, command=copy_to_clipboard).pack(side=LEFT, padx=10)
ttk.Button(gen_buttons, text="➡️ Send to Detector", bootstyle=WARNING, command=import_to_detector).pack(side=LEFT, padx=10)

# --- TAB 2: Email Detection ---
frame_detect = ttk.Frame(notebook)
notebook.add(frame_detect, text="🔍 Detect Email")

ttk.Label(frame_detect, text="Paste Email to Classify", font=("Helvetica", 14, "bold")).pack(pady=10)

input_detect = ttk.Text(frame_detect, height=12, width=100)
input_detect.pack(padx=10, pady=10)

def handle_classify():
    email_text = input_detect.get("1.0", "end").strip()
    if not email_text:
        messagebox.showwarning("Input Needed", "Please enter email text.")
        return
    label, phishing_conf, legit_conf = classify_email(email_text)
    result_msg = (
        f"🧠 Prediction: {label}\n"
        f"🔐 Phishing Confidence: {phishing_conf:.2f}\n"
        f"📩 Legitimate Confidence: {legit_conf:.2f}"
    )
    messagebox.showinfo("Classification Result", result_msg)

ttk.Button(frame_detect, text="🔍 Classify Email", bootstyle=SUCCESS, command=handle_classify).pack(pady=10)

# Start the app
app.mainloop()
