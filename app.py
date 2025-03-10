#!/usr/bin/env python3
import os
import shutil
import glob
import base64
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import csv
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import zipfile
import math

# Page Configuration
st.set_page_config(
    page_title="SFT Model Builder 🚀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model Configuration Class
@dataclass
class ModelConfig:
    name: str
    base_model: str
    size: str
    domain: Optional[str] = None
    
    @property
    def model_path(self):
        return f"models/{self.name}"

# Custom Dataset for SFT
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
        
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        full_text = f"{prompt} {response}"
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = prompt_encoding["input_ids"].squeeze()
        attention_mask = prompt_encoding["attention_mask"].squeeze()
        labels = full_encoding["input_ids"].squeeze()
        
        prompt_len = prompt_encoding["input_ids"].ne(self.tokenizer.pad_token_id).sum().item()
        labels[:prompt_len] = -100  # Mask prompt in loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Model Builder Class
class ModelBuilder:
    def __init__(self):
        self.config = None
        self.model = None
        self.tokenizer = None
        self.sft_data = None

    def load_model(self, model_path: str, config: Optional[ModelConfig] = None):
        with st.spinner("Loading model... ⏳"):
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if config:
                self.config = config
        st.success("Model loaded! ✅")
        return self

    def fine_tune_sft(self, csv_path: str, epochs: int = 3, batch_size: int = 4):
        self.sft_data = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sft_data.append({"prompt": row["prompt"], "response": row["response"]})

        dataset = SFTDataset(self.sft_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        for epoch in range(epochs):
            with st.spinner(f"Training epoch {epoch + 1}/{epochs}... ⚙️"):
                total_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                st.write(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(dataloader):.4f}")
        st.success("SFT Fine-tuning completed! 🎉")
        return self

    def save_model(self, path: str):
        with st.spinner("Saving model... 💾"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        st.success(f"Model saved at {path}! ✅")

    def evaluate(self, prompt: str):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Utility Functions
def get_download_link(file_path, mime_type="text/plain", label="Download"):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{label} 📥</a>'

def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(directory_path))
                zipf.write(file_path, arcname)

def get_model_files():
    return [d for d in glob.glob("models/*") if os.path.isdir(d)]

# Cargo Travel Time Tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: float = 750.0
) -> float:
    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)
    EARTH_RADIUS_KM = 6371.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c
    actual_distance = distance * 1.1
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0
    return round(flight_time, 2)

# Main App
st.title("SFT Model Builder 🤖🚀")

# Sidebar for Model Management
st.sidebar.header("Model Management 🗂️")
model_dirs = get_model_files()
selected_model = st.sidebar.selectbox("Select Saved Model", ["None"] + model_dirs)

if selected_model != "None" and st.sidebar.button("Load Model 📂"):
    if 'builder' not in st.session_state:
        st.session_state['builder'] = ModelBuilder()
    config = ModelConfig(name=os.path.basename(selected_model), base_model="unknown", size="small", domain="general")
    st.session_state['builder'].load_model(selected_model, config)
    st.session_state['model_loaded'] = True
    st.rerun()

# Main UI with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Build New Model 🌱", "Fine-Tune Model 🔧", "Test Model 🧪", "Agentic RAG Demo 🌐"])

with tab1:
    st.header("Build New Model 🌱")
    base_model = st.selectbox(
        "Select Base Model",
        [
            "HuggingFaceTB/SmolLM-135M",  # ~270 MB
            "HuggingFaceTB/SmolLM-360M",  # ~720 MB
            "Qwen/Qwen1.5-0.5B-Chat",     # ~1 GB
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~2 GB, slightly over but included
        ],
        help="Choose a tiny, open-source model (<1 GB except TinyLlama)"
    )
    model_name = st.text_input("Model Name", f"new-model-{int(time.time())}")
    domain = st.text_input("Target Domain", "general")

    if st.button("Download Model ⬇️"):
        config = ModelConfig(name=model_name, base_model=base_model, size="small", domain=domain)
        builder = ModelBuilder()
        builder.load_model(base_model, config)
        builder.save_model(config.model_path)
        st.session_state['builder'] = builder
        st.session_state['model_loaded'] = True
        st.success(f"Model downloaded and saved to {config.model_path}! 🎉")
        st.rerun()

with tab2:
    st.header("Fine-Tune Model 🔧")
    if 'builder' not in st.session_state or not st.session_state.get('model_loaded', False):
        st.warning("Please download or load a model first! ⚠️")
    else:
        if st.button("Generate Sample CSV 📝"):
            sample_data = [
                {"prompt": "What is AI?", "response": "AI is artificial intelligence, simulating human intelligence in machines."},
                {"prompt": "Explain machine learning", "response": "Machine learning is a subset of AI where models learn from data."},
                {"prompt": "What is a neural network?", "response": "A neural network is a model inspired by the human brain."},
            ]
            csv_path = f"sft_data_{int(time.time())}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
                writer.writeheader()
                writer.writerows(sample_data)
            st.markdown(get_download_link(csv_path, "text/csv", "Download Sample CSV"), unsafe_allow_html=True)
            st.success(f"Sample CSV generated as {csv_path}! ✅")

        uploaded_csv = st.file_uploader("Upload CSV for SFT", type="csv")
        if uploaded_csv and st.button("Fine-Tune with Uploaded CSV 🔄"):
            csv_path = f"uploaded_sft_data_{int(time.time())}.csv"
            with open(csv_path, "wb") as f:
                f.write(uploaded_csv.read())
            new_model_name = f"{st.session_state['builder'].config.name}-sft-{int(time.time())}"
            new_config = ModelConfig(
                name=new_model_name,
                base_model=st.session_state['builder'].config.base_model,
                size="small",
                domain=st.session_state['builder'].config.domain
            )
            st.session_state['builder'].config = new_config
            with st.status("Fine-tuning model... ⏳", expanded=True) as status:
                st.session_state['builder'].fine_tune_sft(csv_path)
                st.session_state['builder'].save_model(new_config.model_path)
                status.update(label="Fine-tuning completed! 🎉", state="complete")
            
            zip_path = f"{new_config.model_path}.zip"
            zip_directory(new_config.model_path, zip_path)
            st.markdown(get_download_link(zip_path, "application/zip", "Download Fine-Tuned Model"), unsafe_allow_html=True)
            st.rerun()

with tab3:
    st.header("Test Model 🧪")
    if 'builder' not in st.session_state or not st.session_state.get('model_loaded', False):
        st.warning("Please download or load a model first! ⚠️")
    else:
        if st.session_state['builder'].sft_data:
            st.write("Testing with SFT Data:")
            for item in st.session_state['builder'].sft_data[:3]:
                prompt = item["prompt"]
                expected = item["response"]
                generated = st.session_state['builder'].evaluate(prompt)
                st.write(f"**Prompt**: {prompt}")
                st.write(f"**Expected**: {expected}")
                st.write(f"**Generated**: {generated}")
                st.write("---")

        test_prompt = st.text_area("Enter Test Prompt", "What is AI?")
        if st.button("Run Test ▶️"):
            result = st.session_state['builder'].evaluate(test_prompt)
            st.write(f"**Generated Response**: {result}")

        if st.button("Export Model Files 📦"):
            config = st.session_state['builder'].config
            app_code = f"""
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{config.model_path}")
tokenizer = AutoTokenizer.from_pretrained("{config.model_path}")

st.title("SFT Model Demo")
input_text = st.text_area("Enter prompt")
if st.button("Generate"):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.95, temperature=0.7)
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
            with open("sft_app.py", "w") as f:
                f.write(app_code)
            reqs = "streamlit\ntorch\ntransformers\n"
            with open("sft_requirements.txt", "w") as f:
                f.write(reqs)
            readme = f"""
# SFT Model Demo

## How to run
1. Install requirements: `pip install -r sft_requirements.txt`
2. Run the app: `streamlit run sft_app.py`
3. Input a prompt and click "Generate".
"""
            with open("sft_README.md", "w") as f:
                f.write(readme)
            
            st.markdown(get_download_link("sft_app.py", "text/plain", "Download App"), unsafe_allow_html=True)
            st.markdown(get_download_link("sft_requirements.txt", "text/plain", "Download Requirements"), unsafe_allow_html=True)
            st.markdown(get_download_link("sft_README.md", "text/markdown", "Download README"), unsafe_allow_html=True)
            st.success("Model files exported! ✅")

with tab4:
    st.header("Agentic RAG Demo 🌐")
    st.write("This demo uses tiny models with Agentic RAG to plan a luxury superhero-themed party, enhancing retrieval with DuckDuckGo.")

    if st.button("Run Agentic RAG Demo 🎉"):
        try:
            from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool

            # Load selected tiny model
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

            # Define Agentic RAG agent
            agent = CodeAgent(
                model=model,
                tokenizer=tokenizer,
                tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
                additional_authorized_imports=["pandas"],
                planning_interval=5,
                verbosity_level=2,
                max_steps=15,
            )
            
            task = """
Plan a luxury superhero-themed party at Wayne Manor (42.3601° N, 71.0589° W). Search for the latest superhero party trends using DuckDuckGo,
refine results to include luxury elements (decorations, entertainment, catering), and calculate cargo travel times from key locations 
(e.g., New York, LA, London) to Wayne Manor. Synthesize a complete plan and return it as a pandas dataframe with at least 6 entries 
including locations, travel times, and luxury party ideas.
"""
            with st.spinner("Running Agentic RAG system... ⏳"):
                result = agent.run(task)
            st.write("Agentic RAG Result:")
            st.write(result)
        except ImportError:
            st.error("Please install required packages: `pip install smolagents pandas`")
        except Exception as e:
            st.error(f"Error running demo: {str(e)}")
