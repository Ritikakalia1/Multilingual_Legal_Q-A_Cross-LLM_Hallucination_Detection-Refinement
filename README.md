# Legal AI Platform 🏛️

Multilingual Indian Legal AI System with RAG and Draft Auditing.

## Features
- 🌐 Multilingual Q&A (English, Hindi, Punjabi, Nepali)
- 📁 Law Firm PDF Case File RAG
- ⚖️ Automated Legal Draft Auditor
- 🔄 IPC → BNS Citation Checker

## Setup

### 1. Clone repo
git clone https://github.com/YOUR_USERNAME/legal-llm-platform.git
cd legal-llm-platform

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download adapters from Kaggle
kaggle datasets download YOUR_USERNAME/adapter-en -p adapters/
kaggle datasets download YOUR_USERNAME/adapter-hi -p adapters/
kaggle datasets download YOUR_USERNAME/adapter-pa -p adapters/
kaggle datasets download YOUR_USERNAME/adapter-ne -p adapters/

### 4. Run
python app.py

## Architecture
- Base Model: Qwen2.5-3B-Instruct
- Fine-tuning: QLoRA (4-bit, LoRA r=16)
- Languages: EN, HI, PA, NE
- Legal Acts: IPC, CRPC, Constitution, BNS