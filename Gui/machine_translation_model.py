import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------
# Load trained model
# ------------------------
model = tf.keras.models.load_model(r"C:\Users\Asus\Downloads\model.h5")

# ------------------------
# Load dataset (English-German pairs)
# ------------------------
dataset = load(open(r"C:\Users\Asus\Downloads\english-german-both.pkl", "rb"))

# Split into English / German
english_sentences = dataset[:, 0]
german_sentences = dataset[:, 1]

# ------------------------
# Tokenizers
# ------------------------
german_tokenizer = Tokenizer()
german_tokenizer.fit_on_texts(german_sentences)

english_tokenizer = Tokenizer()
english_tokenizer.fit_on_texts(english_sentences)

# Max lengths
max_german_len = max(len(s.split()) for s in german_sentences)
max_english_len = max(len(s.split()) for s in english_sentences)

# Reverse dict for English (id → word)
reverse_eng_index = {v: k for k, v in english_tokenizer.word_index.items()}

# ------------------------
# Helper Functions
# ------------------------
def preprocess_text(german_text):
    """Convert German sentence → padded sequence"""
    seq = german_tokenizer.texts_to_sequences([german_text])
    seq = pad_sequences(seq, maxlen=max_german_len, padding="post")
    return seq

def decode_prediction(pred):
    """Turn model output → English sentence"""
    pred_ids = np.argmax(pred[0], axis=-1)  # predicted word indices
    words = [reverse_eng_index.get(i, "") for i in pred_ids]
    return " ".join([w for w in words if w])  # remove empty tokens

# ------------------------
# Translate Function
# ------------------------
def translate_sentence():
    german_text = entry.get("1.0", tk.END).strip()
    if not german_text:
        messagebox.showwarning("Input Error", "Please enter a German sentence.")
        return
    
    processed_input = preprocess_text(german_text)
    prediction = model.predict(processed_input)
    english_text = decode_prediction(prediction)
    
    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, english_text)
    output_text.config(state="disabled")

# ------------------------
# GUI Design
# ------------------------
root = tk.Tk()
root.title("🌍 German → English Translator")
root.geometry("650x400")
root.config(bg="#f0f4f7")

# Title
title_label = tk.Label(root, text="German → English Translator", 
                       font=("Helvetica", 20, "bold"), 
                       bg="#2c3e50", fg="white", pady=10)
title_label.pack(fill="x")

# Input Frame
frame_input = tk.Frame(root, bg="#f0f4f7")
frame_input.pack(pady=15)

tk.Label(frame_input, text="Enter German Sentence:", 
         font=("Arial", 12), bg="#f0f4f7").pack(anchor="w")

entry = tk.Text(frame_input, width=70, height=5, font=("Arial", 12))
entry.pack()

# Translate Button
translate_btn = tk.Button(root, text="Translate", command=translate_sentence, 
                          font=("Arial", 14, "bold"), bg="#3498db", fg="white",
                          relief="flat", padx=20, pady=5)
translate_btn.pack(pady=10)

# Output Frame
frame_output = tk.Frame(root, bg="#f0f4f7")
frame_output.pack(pady=10)

tk.Label(frame_output, text="English Translation:", 
         font=("Arial", 12), bg="#f0f4f7").pack(anchor="w")

output_text = tk.Text(frame_output, width=70, height=5, font=("Arial", 12), fg="green")
output_text.pack()
output_text.config(state="disabled")

# Footer
footer = tk.Label(root, text="✨ Built with Tkinter", 
                  font=("Arial", 10, "italic"), bg="#f0f4f7", fg="gray")
footer.pack(side="bottom", pady=5)

root.mainloop()



    
    
    
    
    
    





















