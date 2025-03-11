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

# [Previous sections like ModelConfig, SFTDataset, ModelBuilder, Utility Functions remain unchanged...]

# Cargo Travel Time Tool (Now a Proper smolagents Tool)
from smolagents import tool

@tool
def calculate_cargo_travel_time(origin_coords: Tuple[float, float], destination_coords: Tuple[float, float], cruising_speed_kmh: float = 750.0) -> float:
    """Calculate cargo plane travel time between two coordinates."""
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

# Sidebar with Galleries (unchanged)
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

# Main UI with Tabs (only Tab 4 updated here)
tab1, tab2, tab3, tab4 = st.tabs(["Build Tiny Titan 🌱", "Fine-Tune Titan 🔧", "Test Titan 🧪", "Agentic RAG Party 🌐"])

# [Tab 1, Tab 2, Tab 3 remain unchanged...]

with tab4:
    st.header("Agentic RAG Party 🌐 (Party Like It’s 2099!)")
    st.write("This demo uses Tiny Titans with Agentic RAG to plan a superhero party, powered by DuckDuckGo retrieval!")

    if st.button("Run Agentic RAG Demo 🎉"):
        try:
            from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
            from transformers import AutoModelForCausalLM

            # Load the model
            with st.spinner("Loading SmolLM-135M... ⏳ (Titan’s suiting up!)"):
                model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
                st.write("Model loaded! 🦸‍♂️ (Ready to party!)")

            # Initialize agent with proper tools
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