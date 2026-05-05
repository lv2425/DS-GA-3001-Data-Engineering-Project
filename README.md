# 🏷️ NYC AutoDDG: NYC Automated Dataset Description Generation using Large Language Models

<div align="center">
  <p>
    <img src="https://img.shields.io/static/v1?label=RUFF&message=lint%2Fformat&color=9C27B0&style=flat-square&logo=ruff&logoColor=white" alt="Ruff">
    <img src="https://img.shields.io/badge/Black-formatted-000000?style=flat-square&logo=python&logoColor=white" alt="Black formatted">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10">
    <img src="https://img.shields.io/badge/OpenAI-Model-blue?style=flat-square&logo=openai" alt="OpenAI">
    <img src="https://img.shields.io/badge/Local-LLM-green?style=flat-square&logo=huggingface" alt="Local LLM">
  </p>
</div>

## Overview

This repository contains a tailored implementation based on the **SIGMOD 2026** paper:

> **AutoDDG: Automated Dataset Description Generation using Large Language Models**

This project adapts AutoDDG for **NYC Open Data**, modifying the original framework to take information directly from NYC Open Data. Given a URL, the system  retrieves the corresponding dataset and its metadata, then leverages AutoDDG's data-driven summarization combined with large language models (LLMs) to generate comprehensive, accurate, readable, and concise dataset descriptions. Like the original framework, this implementation supports both API-based (OpenAI) and local LLM (transformers) modes, providing flexibility for different deployment scenarios.

## Installation

Clone the repository and install dependencies via requirements.txt:

```bash
git clone https://github.com/lv2425/DS-GA-3001-Data-Engineering-Project.git
cd AutoDDG
pip install -r requirements.txt
```


**For local LLM support** (Qwen, Llama, etc.), install with optional dependencies:
```bash
pip install git+https://github.com/VIDA-NYU/AutoDDG@main transformers torch
```

---

## Getting Started

AutoDDG supports both **API-based** (OpenAI) and **local LLM** (transformers) modes.

### Using OpenAI API

The simplest way to use AutoDDG is with an OpenAI API client:

```python
import os
import sys
from openai import OpenAI
sys.path.insert(0, "")  # Change your AutoDDG src path here
from autoddg import AutoDDG

# Setup OpenAI client
client = OpenAI(api_key="sk-...")

# Initialize AutoDDG
autoddg = AutoDDG(client=client, model_name="gpt-4o-mini")

# Generate description from a small CSV sample
sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)

print(description)
# >>> This dataset contains medical information about patients, including their unique Case_ID, Age, and Body Mass Index (BMI). etc.
```

### Using Local LLM

AutoDDG also supports local LLMs via transformers (Qwen, Llama, etc.):

```python
import os
import sys
sys.path.insert(0, "")  # Change your AutoDDG src path here
from autoddg import AutoDDG

# Initialize AutoDDG with local LLM
autoddg = AutoDDG(
    client=None,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",  # or any HuggingFace model
    use_local_llm=True,
    local_llm_device="cuda",  # or "cpu" if no GPU
    local_llm_dtype="bfloat16",  # or "float16", "float32"
)

# Generate description
sample_csv = """Case_ID,Age,BMI
C3L-00004,72,22.8
C3L-00010,30,34.15
"""

prompt, description = autoddg.describe_dataset(dataset_sample=sample_csv)
print(description)
```

### Quick Jupyter Notebook Start

For a much better introduction, we **highly recommend** starting with the [quick_start notebook with an example dataset](./examples/quick_start.ipynb).
