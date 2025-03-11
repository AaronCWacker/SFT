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
from PIL import Image
import random
import logging

# Set up logging for feedback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration with Humor
st.set_page_config(
    page_title="SFT Tiny Titans 🚀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/awacke1',
        'Report a bug': 'https://huggingface.co/spaces/awacke1',
        'About': "Tiny Titans: Small models, big dreams, and a sprinkle of chaos! 🌌"
    }
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
        
        full_text = f"{prompt} {response}"
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = full_encoding["input_ids"].squeeze()
        attention_mask = full_encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        prompt_len = prompt_encoding["input_ids"].shape[1]
        if prompt_len < self.max_length:
            labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Model Builder Class with Easter Egg Jokes
class ModelBuilder:
    def __init__(self):
        self.config = None
        self.model = None
        self.tokenizer = None
        self.sft_data = None
        self.jokes = ["Why did the AI go to therapy? Too many layers to unpack! 😂", "Training complete! Time for a binary coffee break. ☕"]

    def load_model(self, model_path: str, config: Optional[ModelConfig] = None):
        with st.spinner(f"Loading {model_path}... ⏳ (Patience, young padawan!)"):
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if config:
                self.config = config
        st.success(f"Model loaded! 🎉 {random.choice(self.jokes)}")
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
            with st.spinner(f"Training epoch {epoch + 1}/{epochs}... ⚙️ (The AI is lifting weights!)"):
                total_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    assert input_ids.shape[0] == labels.shape[0], f"Batch size mismatch: input_ids {input_ids.shape}, labels {labels.shape}"
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                st.write(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(dataloader):.4f}")
        st.success(f"SFT Fine-tuning completed! 🎉 {random.choice(self.jokes)}")
        return self

    def save_model(self, path: str):
        with st.spinner("Saving model... 💾 (Packing the AI’s suitcase!)"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        st.success(f"Model saved at {path}! ✅ May the force be with it.")

    def evaluate(self, prompt: str, status_container=None):
        """Evaluate with feedback"""
        self.model.eval()
        if status_container:
            status_container.write("Preparing to evaluate... 🧠 (Titan’s warming up its circuits!)")
        logger.info(f"Evaluating prompt: {prompt}")
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(self.model.device)
                if status_container:
                    status_container.write(f"Tokenized input shape: {inputs['input_ids'].shape} 📏")
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7
                )
                if status_container:
                    status_container.write("Generation complete! Decoding response... 🗣")
                
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Generated response: {result}")
                return result
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            if status_container:
                status_container.error(f"Oops! Something broke: {str(e)} 💥 (Titan tripped over a wire!)")
            return f"Error: {str(e)}"

# Utility Functions with Wit
def get_download_link(file_path, mime_type="text/plain", label="Download"):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{label} 📥 (Grab it before it runs away!)</a>'

def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(directory_path))
                zipf.write(file_path, arcname)

def get_model_files():
    return [d for d in glob.glob("models/*") if os.path.isdir(d)]

def get_gallery_files(file_types):
    files = []
    for ext in file_types:
        files.extend(glob.glob(f"*.{ext}"))
    return sorted(files)

# Cargo Travel Time Tool
def calculate_cargo_travel_time(origin_coords: Tuple[float, float], destination_coords: Tuple[float, float], cruising_speed_kmh: float = 750.0) -> float:
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
st.title("SFT Tiny Titans 🚀 (Small but Mighty!)")

# Sidebar with Galleries
st.sidebar.header("Galleries & Shenanigans 🎨")
st.sidebar.subheader("Image Gallery 📸")
img_files = get_gallery_files(["png", "jpg", "jpeg"])
if img_files:
    img_cols = st.sidebar.slider("Image Columns 📸", 1, 5, 3)
    cols = st.sidebar.columns(img_cols)
    for idx, img_file in enumerate(img_files[:img_cols * 2]):
        with cols[idx % img_cols]:
            st.image(Image.open(img_file), caption=f"{img_file} 🖼", use_column_width=True)

st.sidebar.subheader("CSV Gallery 📊")
csv_files = get_gallery_files(["csv"])
if csv_files:
    for csv_file in csv_files[:5]:
        st.sidebar.markdown(get_download_link(csv_file, "text/csv", f"{csv_file} 📊"), unsafe_allow_html=True)

st.sidebar.subheader("Model Management 🗂️")
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
tab1, tab2, tab3, tab4 = st.tabs(["Build Tiny Titan 🌱", "Fine-Tune Titan 🔧", "Test Titan 🧪", "Agentic RAG Party 🌐"])

with tab1:
    st.header("Build Tiny Titan 🌱 (Assemble Your Mini-Mecha!)")
    base_model = st.selectbox(
        "Select Tiny Model",
        ["HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M", "Qwen/Qwen1.5-0.5B-Chat"],
        help="Pick a pint-sized powerhouse (<1 GB)! SmolLM-135M (~270 MB), SmolLM-360M (~720 MB), Qwen1.5-0.5B (~1 GB)"
    )
    model_name = st.text_input("Model Name", f"tiny-titan-{int(time.time())}")
    domain = st.text_input("Target Domain", "general")

    if st.button("Download Model ⬇️"):
        config = ModelConfig(name=model_name, base_model=base_model, size="small", domain=domain)
        builder = ModelBuilder()
        builder.load_model(base_model, config)
        builder.save_model(config.model_path)
        st.session_state['builder'] = builder
        st.session_state['model_loaded'] = True
        st.success(f"Model downloaded and saved to {config.model_path}! 🎉 (Tiny but feisty!)")
        st.rerun()

with tab2:
    st.header("Fine-Tune Titan 🔧 (Teach Your Titan Some Tricks!)")
    if 'builder' not in st.session_state or not st.session_state.get('model_loaded', False):
        st.warning("Please build or load a Titan first! ⚠️ (No Titan, no party!)")
    else:
        if st.button("Generate Sample CSV 📝"):
            sample_data = [
                {"prompt": "What is AI?", "response": "AI is artificial intelligence, simulating human smarts in machines."},
                {"prompt": "Explain machine learning", "response": "Machine learning is AI’s gym where models bulk up on data."},
                {"prompt": "What is a neural network?", "response": "A neural network is a brainy AI mimicking human noggins."},
            ]
            csv_path = f"sft_data_{int(time.time())}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
                writer.writeheader()
                writer.writerows(sample_data)
            st.markdown(get_download_link(csv_path, "text/csv", "Download Sample CSV"), unsafe_allow_html=True)
            st.success(f"Sample CSV generated as {csv_path}! ✅ (Fresh from the data oven!)")

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
            with st.status("Fine-tuning Titan... ⏳ (Whipping it into shape!)", expanded=True) as status:
                st.session_state['builder'].fine_tune_sft(csv_path)
                st.session_state['builder'].save_model(new_config.model_path)
                status.update(label="Fine-tuning completed! 🎉 (Titan’s ready to rumble!)", state="complete")
            
            zip_path = f"{new_config.model_path}.zip"
            zip_directory(new_config.model_path, zip_path)
            st.markdown(get_download_link(zip_path, "application/zip", "Download Fine-Tuned Titan"), unsafe_allow_html=True)
            st.rerun()

with tab3:
    st.header("Test Titan 🧪 (Put Your Titan to the Test!)")
    if 'builder' not in st.session_state or not st.session_state.get('model_loaded', False):
        st.warning("Please build or load a Titan first! ⚠️ (No Titan, no test drive!)")
    else:
        if st.session_state['builder'].sft_data:
            st.write("Testing with SFT Data:")
            with st.spinner("Running SFT data tests... ⏳ (Titan’s flexing its brain muscles!)"):
                for item in st.session_state['builder'].sft_data[:3]:
                    prompt = item["prompt"]
                    expected = item["response"]
                    status_container = st.empty()
                    generated = st.session_state['builder'].evaluate(prompt, status_container)
                    st.write(f"**Prompt**: {prompt}")
                    st.write(f"**Expected**: {expected}")
                    st.write(f"**Generated**: {generated} (Titan says: '{random.choice(['Bleep bloop!', 'I am groot!', '42!'])}')")
                    st.write("---")
                    status_container.empty()  # Clear status after each test

        test_prompt = st.text_area("Enter Test Prompt", "What is AI?")
        if st.button("Run Test ▶️"):
            with st.spinner("Testing your prompt... ⏳ (Titan’s pondering deeply!)"):
                status_container = st.empty()
                result = st.session_state['builder'].evaluate(test_prompt, status_container)
                st.write(f"**Generated Response**: {result} (Titan’s wisdom unleashed!)")
                status_container.empty()

        if st.button("Export Titan Files 📦"):
            config = st.session_state['builder'].config
            app_code = f"""
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{config.model_path}")
tokenizer = AutoTokenizer.from_pretrained("{config.model_path}")

st.title("Tiny Titan Demo")
input_text = st.text_area("Enter prompt")
if st.button("Generate"):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.95, temperature=0.7)
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
            with open("titan_app.py", "w") as f:
                f.write(app_code)
            reqs = "streamlit\ntorch\ntransformers\n"
            with open("titan_requirements.txt", "w") as f:
                f.write(reqs)
            readme = f"""
# Tiny Titan Demo

## How to run
1. Install requirements: `pip install -r titan_requirements.txt`
2. Run the app: `streamlit run titan_app.py`
3. Input a prompt and click "Generate". Watch the magic unfold! 🪄
"""
            with open("titan_README.md", "w") as f:
                f.write(readme)
            
            st.markdown(get_download_link("titan_app.py", "text/plain", "Download App"), unsafe_allow_html=True)
            st.markdown(get_download_link("titan_requirements.txt", "text/plain", "Download Requirements"), unsafe_allow_html=True)
            st.markdown(get_download_link("titan_README.md", "text/markdown", "Download README"), unsafe_allow_html=True)
            st.success("Titan files exported! ✅ (Ready to conquer the galaxy!)")

with tab4:
    st.header("Agentic RAG Party 🌐 (Party Like It’s 2099!)")
    st.write("This demo uses Tiny Titans with Agentic RAG to plan a superhero party, powered by DuckDuckGo retrieval!")

    if st.button("Run Agentic RAG Demo 🎉"):
        try:
            from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
            from transformers import AutoModelForCausalLM

            # Load the model without separate tokenizer for agent
            with st.spinner("Loading SmolLM-135M... ⏳ (Titan’s suiting up!)"):
                model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
                st.write("Model loaded! 🦸‍♂️ (Ready to party!)")

            # Initialize agent without tokenizer argument
            agent = CodeAgent(
                model=model,
                tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
                additional_authorized_imports=["pandas"],
                planning_interval=5,
                verbosity_level=2,
                max_steps=15,
            )
            
            task = """
Plan a luxury superhero-themed party at Wayne Manor (42.3601° N, 71.0589° W). Use DuckDuckGo to search for the latest superhero party trends,
refine results for luxury elements (decorations, entertainment, catering), and calculate cargo travel times from key locations 
(New York: 40.7128° N, 74.0060° W; LA: 34.0522° N, 118.2437° W; London: 51.5074° N, 0.1278° W) to Wayne Manor. 
Synthesize a plan with at least 6 entries in a pandas dataframe, including locations, travel times, and luxury ideas.
Add a random superhero catchphrase to each entry for fun!
"""
            with st.spinner("Planning the ultimate superhero bash... ⏳ (Calling all caped crusaders!)"):
                result = agent.run(task)
                st.write("Agentic RAG Party Plan:")
                st.write(result)
                st.write("Party on, Wayne! 🦸‍♂️🎉")
        except ImportError:
            st.error("Please install required packages: `pip install smolagents pandas transformers`")
        except TypeError as e:
            st.error(f"Agent setup failed: {str(e)} (Looks like the Titans need a tune-up!)")
        except Exception as e:
            st.error(f"Error running demo: {str(e)} (Even Batman has off days!)")
