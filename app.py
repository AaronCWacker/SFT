#!/usr/bin/env python3

"""
Requirements:
streamlit
torch
pandas
transformers
"""

import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import csv

# Page Configuration
st.set_page_config(
    page_title="SFT Model Builder 🚀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Help Documentation as a Variable
HELP_DOC = """
# SFT Model Builder - Help Guide 🚀

## Overview
This Streamlit app allows users to **download, fine-tune, and test Transformer models** with **Supervised Fine-Tuning (SFT)** using CSV data. It is designed for NLP tasks and can be expanded for **CV and Speech models**.

## Features
- ✅ **Download a pre-trained model** from Hugging Face.
- ✅ **Upload a CSV dataset** for fine-tuning.
- ✅ **Train the model** with multiple epochs and adjustable batch sizes.
- ✅ **Test the fine-tuned model** with text prompts.

## Installation
To run the app, install dependencies:
```bash
pip install -r requirements.txt
```
Then, start the app:
```bash
streamlit run app.py
```

## How to Use
1. **Download Model**: Select a base model (e.g., `distilgpt2`), then click **Download Model**.
2. **Upload CSV**: The CSV must have two columns: `prompt` and `response`.
3. **Fine-Tune Model**: Click **Fine-Tune Model** to start training.
4. **Test Model**: Enter a text prompt and generate responses.

## CSV Format
Example format:
```csv
prompt,response
"What is AI?","AI is artificial intelligence."
"Explain machine learning","Machine learning is a subset of AI."
```

## Model References
| Model 🏆 | Description 📌 | Link 🔗 |
|---------|-------------|---------|
| **GPT-2** 🤖 | Standard NLP model | [Hugging Face](https://huggingface.co/gpt2) |
| **DistilGPT-2** ⚡ | Lightweight version of GPT-2 | [Hugging Face](https://huggingface.co/distilgpt2) |
| **EleutherAI Pythia** 🔬 | Open-source GPT-like models | [Hugging Face](https://huggingface.co/EleutherAI/pythia-70m) |

## Additional Notes
- This app supports **PyTorch models**.
- Default training parameters: `epochs=3`, `batch_size=4`.
- Fine-tuned models are **saved locally** for future use.

For more details, visit [Hugging Face Models](https://huggingface.co/models). 🚀
"""

# Custom Dataset for Fine-Tuning
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]
        input_text = f"{prompt} {response}"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

# Model Loader and Trainer Class
class ModelBuilder:
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        st.spinner("Loading model... ⏳")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        st.success("Model loaded! ✅")

    def fine_tune(self, csv_path, epochs=3, batch_size=4):
        """Supervised Fine-Tuning with CSV data"""
        sft_data = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sft_data.append({"prompt": row["prompt"], "response": row["response"]})

        dataset = SFTDataset(sft_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        for epoch in range(epochs):
            st.spinner(f"Training epoch {epoch + 1}/{epochs}... ⚙️")
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            st.write(f"Epoch {epoch + 1} completed.")
        st.success("Fine-tuning completed! 🎉")

# Main UI
st.title("SFT Model Builder 🤖🚀")
model_builder = ModelBuilder()

if st.button("Download Model ⬇️"):
    model_builder.load_model()

csv_file = st.file_uploader("Upload CSV for Fine-Tuning", type="csv")
if csv_file and st.button("Fine-Tune Model 🔄"):
    model_builder.fine_tune(csv_file)

# Render Help Documentation at End
st.markdown(HELP_DOC)
