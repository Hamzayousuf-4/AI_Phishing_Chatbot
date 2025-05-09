from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ✅ Local model path
model_path = r"C:\Users\Hamza\Desktop\AI_Phishing_Chatbot\generator\final_gpt2"

# ✅ Load model and tokenizer offline
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
model.eval()

# 🧠 Define the prompt (you can also replace this with input())
prompt = "Urgent action required"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# ✨ Generate phishing-style email
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id  # prevent warning
    )

# 📬 Output generated text
generated_email = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\n📨 Prompt: {prompt}")
print("\n📬 Generated Phishing Email:\n")
print(generated_email)
