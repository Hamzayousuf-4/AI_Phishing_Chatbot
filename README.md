# AI Phishing Email Detection and Generation

This project includes a **fine-tuned GPT-2 model** for phishing email generation and a **fine-tuned BERT model** for phishing email detection. It also features a GUI built with `tkinter` and `ttkbootstrap`.

## 💡 Features

- 🧠 Phishing email generation using GPT-2
- 🔍 Email classification using BERT
- 🖥 Desktop GUI for both generation and detection
- 🧪 Support for retraining models using your own dataset

## 📁 Project Structure



chatbot/ # GUI and main logic
detector/ # BERT model and training scripts
generator/ # GPT-2 model and training scripts
data/ # Raw and processed email datasets
utils/ # Cleaning and preprocessing code
requirements.txt # All dependencies
README.md # This file



## 🛠 Setup

1. Clone the repository and open terminal in project root.
2. Create a virtual environment (recommended):

```bash
python -m venv chatbot_env
chatbot_env\Scripts\activate  # Windows


Install required libraries:
pip install -r requirements.txt


🚀 Run the App
python chatbot/chatbot_gui.py


Retrain Models

If you want to retrain:
# GPT-2 Generator
python generator/train_generator.py
# BERT Detector
python detector/train_detector.py

⚠️ Note: The GPT-2 and BERT fine-tuned models are not included due to file size limits. 
You can retrain them using the provided `train_generator.py` and `train_detector.py` scripts 

if you have any questions you can contact me at my www.linkedin.com/in/hamza-yousuf-6a8b1334a account 

Make sure your data is cleaned and located in data/processed/balanced_cleaned_emails.csv

Hamza Yousuf 


