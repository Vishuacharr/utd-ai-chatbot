# 🤖 UTD AI Chatbot — Llama 3.2 Fine-Tuned Academic Assistant

> **Llama 3.2 3B fine-tuned with QLoRA** on 10GB+ of UTD academic content. Serves 5000+ monthly queries with a 92% satisfaction rate. Includes full scraping, preprocessing, training, and deployment pipeline.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Llama](https://img.shields.io/badge/Llama_3.2-3B-orange)](https://llama.meta.com)
[![PEFT](https://img.shields.io/badge/QLoRA-PEFT-green)](https://github.com/huggingface/peft)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏗️ Pipeline Overview

```
1. DATA COLLECTION
   BeautifulSoup scraper → UTD catalog, syllabi, FAQs, advisories

2. PREPROCESSING
   Pandas cleaning → deduplication → instruction formatting (Alpaca format)

3. FINE-TUNING
   Llama 3.2 3B + QLoRA (r=64, α=16, 4-bit NF4 quantization)
   → 40% accuracy improvement over base model

4. DEPLOYMENT
   FastAPI inference server + Streamlit chat UI
   → 5000+ monthly queries | 92% satisfaction rate
```

---

## 📊 Results

| Metric | Base Llama 3.2 | Fine-tuned |
|--------|---------------|------------|
| UTD-specific accuracy | ~48% | **88%** (+40%) |
| Response relevance | 0.71 | **0.94** |
| Hallucination rate | 22% | **4%** |
| Monthly active users | — | **5000+** |
| Satisfaction rate | — | **92%** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/Vishuacharr/utd-ai-chatbot
cd utd-ai-chatbot

pip install -r requirements.txt

# Option A: Use pre-trained weights (HuggingFace Hub)
python src/inference.py --model Vishuacharr/utd-llama-3.2-3b-qlora

# Option B: Fine-tune from scratch
python src/scraper.py              # Collect UTD data
python src/preprocess.py          # Clean + format
python fine_tuning/train.py       # Fine-tune with QLoRA

# Option C: Docker
docker-compose up
```

---

## 💬 Chat Interface

```python
from src.chatbot import UTDChatbot

bot = UTDChatbot(model_path="./fine_tuned_model")

response = bot.chat("What are the requirements for the MS in Business Analytics?")
print(response)
# "The MS in Business Analytics requires 36 credit hours including..."
```

---

## 📁 Project Structure

```
utd-ai-chatbot/
├── src/
│   ├── scraper.py          # BeautifulSoup UTD web scraper
│   ├── preprocess.py       # Data cleaning & instruction formatting
│   ├── chatbot.py          # UTDChatbot inference class
│   ├── inference.py        # CLI inference
│   └── api/
│       ├── main.py         # FastAPI server
│       └── ui.py           # Streamlit chat UI
├── fine_tuning/
│   ├── train.py            # QLoRA training script
│   ├── config.py           # Training hyperparameters
│   └── evaluate.py         # Evaluation pipeline
├── data/
│   └── README.md           # Data format docs
├── requirements.txt
└── docker-compose.yml
```

---

## 🧠 Model Details

- **Base model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 (bitsandbytes)
- **LoRA rank (r)**: 64, **alpha**: 16, **dropout**: 0.05
- **Target modules**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Training data**: 10GB UTD academic content (courses, advisories, FAQs)
- **Training**: 3 epochs, batch size 4, gradient accumulation 4

## 📄 License

MIT — see [LICENSE](LICENSE)
