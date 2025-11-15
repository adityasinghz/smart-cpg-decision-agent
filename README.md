# Smart CPG Decision Support Agent

## Overview
The **Smart CPG Decision Support Agent** uses **Generative AI (GenAI)** and **Agentic AI** to help Consumer Packaged Goods (CPG) businesses analyze sales data and make data-driven decisions.

It ingests multi-store, multi-SKU sales data, detects trends and anomalies, simulates business scenarios, and produces natural-language strategy memos through an agentic reasoning loop.

---

## 📥 Prerequisites & Downloads

### Quick Checklist

Before starting, ensure you have:

- [ ] **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- [ ] **Git** installed ([Download](https://git-scm.com/downloads)) - for cloning repository
- [ ] **Internet connection** - for downloading packages and API access
- [ ] **~2-5 GB free disk space** - for Python packages and dependencies
- [ ] **API Key** (optional but recommended) - Hugging Face (free) or OpenAI (paid)

### Required Software

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python --version` or `python3 --version`
   - ⚠️ **Important (Windows)**: Check "Add Python to PATH" during installation
   - **Recommended**: Python 3.9 or 3.10 for best compatibility

2. **Git** (for cloning the repository)
   - Download from: https://git-scm.com/downloads
   - Verify installation: `git --version`
   - **Alternative**: Download repository as ZIP if you don't need Git

### Required Python Packages

All packages are listed in `requirements.txt` and will be installed automatically with:
```bash
pip install -r requirements.txt
```

**What gets downloaded (~500MB-2GB total):**

| Category | Packages | Size | Purpose |
|----------|----------|------|---------|
| **Data Processing** | pandas, numpy, pyarrow, pyspark | ~200MB | Data manipulation and analysis |
| **UI Framework** | streamlit | ~50MB | Web interface |
| **AI/ML Core** | scikit-learn, transformers, torch | ~1-2GB | Machine learning models (PyTorch is largest) |
| **Visualization** | plotly, matplotlib, seaborn | ~100MB | Charts and graphs |
| **LLM Integration** | openai, huggingface-hub, langchain | ~200MB | AI model access |
| **Utilities** | python-dotenv | ~1MB | Environment variable management |

**Installation Notes:**
- First-time installation: 5-15 minutes (depends on internet speed)
- PyTorch will auto-detect your system (CPU/GPU) and install appropriate version
- If you have GPU, PyTorch will install CUDA-enabled version (~2GB)
- If CPU-only, PyTorch installs CPU version (~500MB)

### Optional: API Keys (Choose ONE)

**Option 1: Hugging Face (FREE - Recommended for Testing)**
- Get free token: https://huggingface.co/settings/tokens
- No credit card required
- Rate limits apply (higher limits with token)

**Option 2: OpenAI (Paid)**
- Get API key: https://platform.openai.com/api-keys
- Requires credit card
- Pay-per-use pricing

**Option 3: Azure OpenAI**
- Requires Azure subscription
- Get endpoint and key from Azure Portal

### Data File

The application requires sales data in parquet or CSV format:
- **Location**: `data/cpg_sales_data.parquet` or `data/cpg_sales_data.parquet.csv`
- **Format**: Parquet (preferred) or CSV
- **Schema**: See "Expected Schema" section below
- **Sample data**: Included in repository or can be generated

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

**Install all required packages:**
```bash
pip install -r requirements.txt
```

**Note**: This will download and install:
- ~500MB-2GB of packages (depending on your system)
- PyTorch (for ML models) - largest package
- Streamlit and visualization libraries
- LangChain and AI framework packages

**Installation time**: 5-15 minutes depending on internet speed

### Step 2: Set Up API Key (Choose ONE Option)

#### Option A: Hugging Face (FREE! ⭐ Recommended for Testing)
```bash
# All Hugging Face dependencies are already in requirements.txt
# Just create .env file (optional - works without token but has rate limits)
echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env
```
Get free token: https://huggingface.co/settings/tokens

#### Option B: OpenAI (Paid, High Quality)
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```
Get your API key: https://platform.openai.com/api-keys

#### Option C: Azure OpenAI
```bash
# Create .env file
echo "AZURE_OPENAI_API_KEY=your-key" > .env
echo "AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/" >> .env
```

### Step 3: Prepare Data

**Option A: Use Provided Sample Data**
- The repository includes `data/cpg_sales_data.parquet.csv`
- If using CSV, the app will automatically load it
- For better performance, convert to parquet:
  ```python
  import pandas as pd
  df = pd.read_csv('data/cpg_sales_data.parquet.csv')
  df.to_parquet('data/cpg_sales_data.parquet', index=False)
  ```

**Option B: Generate New Sample Data**
```bash
python scripts/generate_sample_data.py
```

**Option C: Use Your Own Data**
- Place your data file at `data/cpg_sales_data.parquet` or `data/cpg_sales_data.parquet.csv`
- Ensure it matches the expected schema (see below)

### Step 4: Run the Application
```bash
streamlit run src/ui/streamlit_app.py
```

---

## 📋 Features

- **Automated Data Ingestion:** Load and preprocess large, complex datasets using PySpark or pandas
- **Trend & Anomaly Detection:** Identify seasonality, promotions, and unexpected shifts in sales
- **Scenario Simulation:** Run "what-if" simulations such as price hikes or promo campaigns
- **AI-Generated Insights:** Summarize and explain outcomes in natural language
- **Agentic Loop:** Use LangChain/LangGraph to choose and execute analysis tools automatically
- **Interactive UI:** Streamlit dashboard for conversational insights

---

## 🧱 Project Structure

```
smart-cpg-decision-agent/
├── data/
│   └── cpg_sales_data.parquet          # Sales data (parquet format)
│
├── notebooks/
│   ├── 01_EDA_and_Data_Loading.ipynb
│   ├── 02_Trend_Anomaly_Detection.ipynb
│   ├── 03_Scenario_Simulation.ipynb
│   └── 04_Agent_Loop_Prototype.ipynb
│
├── src/
│   ├── data_loader.py                  # Data loading utilities
│   ├── tools/
│   │   ├── trend_analysis.py          # Trend extraction tools
│   │   ├── anomaly_detection.py       # Anomaly detection tools
│   │   └── scenario_simulation.py     # Scenario simulation tools
│   ├── genai/
│   │   └── llm_interface.py           # LLM interface (OpenAI/Azure/HF)
│   ├── agent/
│   │   ├── agent_core.py              # Main agent logic
│   │   └── memory.py                  # Conversation memory
│   └── ui/
│       ├── streamlit_app.py           # Streamlit web UI
│       └── cli.py                     # Command-line interface
│
├── tests/                              # Unit tests
├── requirements.txt                    # Python dependencies
├── .env                                # API keys (not committed)
└── README.md                           # This file
```

---

## 📦 Dependencies & Imports

### Core Dependencies (requirements.txt)
```
# Core data processing
pyspark>=3.4.0
pandas>=2.0.0
pyarrow>=12.0.0
numpy>=1.24.0

# UI
streamlit>=1.28.0

# LangChain and AI frameworks
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-huggingface>=0.0.1
langchain-community>=0.0.20
langgraph>=0.0.1
langchain-core>=0.1.0

# LLM providers
openai>=1.0.0
huggingface-hub>=0.19.0

# Hugging Face models (required for Hugging Face integration)
transformers>=4.35.0
torch>=2.0.0
accelerate>=0.24.0

# Machine learning
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
python-dotenv>=1.0.0

# Optional (not actively used but listed for completeness)
crewai>=0.1.0
```

### Key Imports Used in Project

**LLM Interface (`src/genai/llm_interface.py`):**
```python
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import torch
```

**Agent Core (`src/agent/agent_core.py`):**
```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
```

**Data Tools:**
```python
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/smart-cpg-decision-agent.git
cd smart-cpg-decision-agent
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

**For Hugging Face (FREE):**
```env
HUGGINGFACE_API_TOKEN=your_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**For OpenAI:**
```env
OPENAI_API_KEY=sk-your-key-here
```

**For Azure OpenAI:**
```env
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 5. Prepare Data

The agent expects data in parquet or CSV format at `data/cpg_sales_data.parquet` or `data/cpg_sales_data.parquet.csv`.

**Expected Schema (Required Columns):**
- `date` - Date/time column
- `store_id` - Store identifier
- `store_region` - Geographic region
- `sku_id` - Product SKU identifier
- `category` - Product category
- `units_sold` - Units sold
- `revenue` - Revenue amount
- `promo_flag` - Promotion indicator (0/1)
- `promo_type` - Type of promotion
- `price` - Product price
- `inventory_level` - Inventory level
- `store_size` - Store size category
- `holiday_flag` - Holiday indicator (0/1)

**Download/Generate Sample Data:**

You can either:
1. Use the provided `data/cpg_sales_data.parquet.csv` file (already in repository)
2. Generate new sample data using the script below
3. Use your own data (must match schema)

**Generate Sample Data Script:**
```python
import pandas as pd
import numpy as np

np.random.seed(123)
num_rows = 5000
date_range = pd.date_range('2022-01-01', periods=365)
stores = list(range(1, 11))
regions = ['North', 'South', 'East']
skus = list(range(101, 151))
categories = ['Beverages', 'Snacks', 'Dairy', 'Household', 'Personal Care']
promo_types = [None, 'Discount', 'BuyOneGetOne', 'FlashSale']
store_sizes = ['Small', 'Medium', 'Large']

data = []
for _ in range(num_rows):
    date = np.random.choice(date_range)
    store = np.random.choice(stores)
    store_region = np.random.choice(regions)
    sku = np.random.choice(skus)
    category = np.random.choice(categories)
    base_price = round(np.random.uniform(2, 10), 2)
    promo_flag = np.random.choice([0, 1], p=[0.8, 0.2])
    promo_type = np.random.choice(promo_types) if promo_flag else None
    price = base_price * (0.8 if promo_flag else 1.0)
    units_sold = np.random.poisson(20) if promo_flag else np.random.poisson(10)
    revenue = units_sold * price
    inventory_level = np.random.randint(100, 1000)
    store_size = np.random.choice(store_sizes)
    holiday_flag = 1 if date.weekday() in [5, 6] else 0
    data.append([date, store, store_region, sku, category, units_sold, revenue,
                 promo_flag, promo_type, price, inventory_level, store_size, holiday_flag])

df = pd.DataFrame(data, columns=[
    'date', 'store_id', 'store_region', 'sku_id', 'category',
    'units_sold', 'revenue', 'promo_flag', 'promo_type',
    'price', 'inventory_level', 'store_size', 'holiday_flag'
])

df.to_parquet('data/cpg_sales_data.parquet', index=False)
print("✅ Synthetic dataset created at data/cpg_sales_data.parquet")
```

---

## 🖥️ Usage

### Streamlit UI (Recommended)

```bash
streamlit run src/ui/streamlit_app.py
```

The app will open in your browser. You can:
- Select Hugging Face (FREE) or OpenAI
- Load your data file
- Ask questions about sales trends, anomalies, and scenarios
- View AI-generated strategy memos
- See analysis results and tool usage

**Example Questions:**
- "What are the sales trends for the last quarter?"
- "Compare performance across different stores"
- "What would happen if we run a 15% discount promotion?"
- "Detect any anomalies in sales data"
- "What are the seasonal patterns in our sales?"

### CLI Interface

```bash
# Interactive mode
python -m src.ui.cli --data data/cpg_sales_data.parquet

# Single question mode
python -m src.ui.cli --data data/cpg_sales_data.parquet --question "What are the sales trends?"

# Use Azure OpenAI
python -m src.ui.cli --data data/cpg_sales_data.parquet --azure
```

### Python API

```python
from src.agent.agent_core import CPGDecisionAgent
from src.genai.llm_interface import LLMInterface
from src.agent.memory import SessionMemory

# Option 1: Hugging Face (FREE)
llm = LLMInterface(
    use_huggingface=True,
    huggingface_model="mistralai/Mistral-7B-Instruct-v0.2"
)

# Option 2: OpenAI
llm = LLMInterface(model="gpt-4")

# Option 3: Azure OpenAI
llm = LLMInterface(
    model="gpt-4",
    use_azure=True
)

# Initialize agent
memory = SessionMemory()
agent = CPGDecisionAgent(
    llm=llm,
    memory=memory,
    data_path="data/cpg_sales_data.parquet"
)

# Ask a question
result = agent.run("What are the sales trends for the last quarter?")
print(result['response'])
print(result['strategy_memo'])
```

---

## 🧠 Agentic Architecture

| Layer | Purpose |
|-------|----------|
| **Data Layer** | Load, clean, and validate historical sales data |
| **Tool Layer** | Trend, anomaly, and simulation utilities |
| **GenAI Layer** | Use OpenAI / Azure OpenAI / Hugging Face for generating insights |
| **Agent Layer** | Manage tool orchestration and context (LangChain/LangGraph) |
| **UI Layer** | Streamlit and CLI interfaces for interaction |

**Flow Example:**
1. User asks, "What if we apply a 10% discount on Beverages in the North region?"
2. Agent picks the right simulation tools
3. Simulation produces new KPIs
4. LLM generates a summary memo with recommendations
5. UI displays graphs and text insights

### Available Tools

The agent has access to these tools:
- **extract_trends** - Analyze sales trends over time
- **detect_seasonality** - Find seasonal patterns in sales
- **detect_anomalies** - Identify unusual sales patterns
- **compare_stores** - Compare performance across stores
- **simulate_promotion** - Simulate promotional campaigns
- **simulate_price_change** - Simulate price increases/decreases
- **get_data_summary** - Get dataset statistics and overview

---

## 🤗 Hugging Face Setup (Free Alternative)

### Quick Setup

**Step 1: Get Free Token (Optional but Recommended)**
1. Go to https://huggingface.co/settings/tokens
2. Sign up (free)
3. Create a new token (read access is enough)
4. Copy the token

**Step 2: Install Dependencies**
```bash
# All dependencies are in requirements.txt
pip install -r requirements.txt
```

**Step 3: Set Up `.env` File**
```env
HUGGINGFACE_API_TOKEN=your_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**Step 4: Use in Code**
```python
from src.genai.llm_interface import LLMInterface

llm = LLMInterface(
    use_huggingface=True,
    huggingface_model="mistralai/Mistral-7B-Instruct-v0.2"
)
```

### Recommended Models

**For Inference API (Free Tier):**
- `mistralai/Mistral-7B-Instruct-v0.2` - Fast, good quality ⭐
- `microsoft/phi-2` - Very small, fast
- `google/flan-t5-large` - Smaller, faster
- `meta-llama/Llama-2-7b-chat-hf` - Requires access request

**For Local Models (If You Have GPU):**
- `mistralai/Mistral-7B-Instruct-v0.2` - Best balance
- `microsoft/phi-2` - Smallest, works on CPU

### Comparison: OpenAI vs Hugging Face

| Feature | OpenAI | Hugging Face (API) | Hugging Face (Local) |
|---------|--------|-------------------|---------------------|
| **Cost** | Pay-per-use | Free (rate limits) | Free |
| **Setup** | API key | Token (optional) | Install packages |
| **Speed** | Fast | Medium | Depends on hardware |
| **Quality** | Excellent | Good | Good |
| **Privacy** | Data sent to OpenAI | Data sent to HF | Fully local |
| **GPU Required** | No | No | Recommended |

---

## 🔐 Environment Variables

Create a `.env` file in the project root (never commit it):

```env
# Hugging Face (FREE - Recommended for testing)
HUGGINGFACE_API_TOKEN=your_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# OpenAI (Paid)
OPENAI_API_KEY=sk-your-key-here

# Azure OpenAI (For Azure deployments)
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Data path (optional)
DATA_PATH=data/cpg_sales_data.parquet
```

---

## 🧰 .gitignore

The following are already in `.gitignore`:
```
__pycache__/
*.pyc
.venv/
.env
data/*.parquet
data/*.csv
.ipynb_checkpoints
.DS_Store
Thumbs.db
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent.py
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Error: "API key not found"
- Make sure `.env` file exists in project root
- Check that the key name is exactly `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, or `HUGGINGFACE_API_TOKEN`
- Verify the key is not wrapped in quotes in `.env` file

### Error: "Hugging Face packages required"
```bash
# All Hugging Face dependencies are in requirements.txt
pip install -r requirements.txt
```

### Error: "Rate limit exceeded" (Hugging Face)
- Get a free token at https://huggingface.co/settings/tokens
- Add to `.env`: `HUGGINGFACE_API_TOKEN=your_token`
- Or wait a few minutes and try again

### Error: "Data file not found"
- Make sure `data/cpg_sales_data.parquet` exists
- Or generate sample data: `python scripts/generate_sample_data.py`
- Or load data manually in the Streamlit UI

### Error: "Model not supported for task"
- For Mistral models, the system automatically uses conversational API
- If issues persist, try a different model like `microsoft/phi-2`

---

## ☁️ Databricks Integration

1. Upload your dataset (`.csv` or `.parquet`) to **Azure Databricks**
2. Connect this repository to **Databricks Repos**
3. Run the notebooks in sequence:
   - `01_EDA_and_Data_Loading.ipynb`
   - `02_Trend_Anomaly_Detection.ipynb`
   - `03_Scenario_Simulation.ipynb`
   - `04_Agent_Loop_Prototype.ipynb`

Example snippet:
```python
spark_df = spark.read.parquet("/dbfs/mnt/data/cpg_sales_data.parquet")
display(spark_df.limit(100))
```

---

## 🔮 Future Enhancements

- Connect to **Azure Data Lake** for large-scale ingestion  
- Add **forecasting models** (Prophet / ARIMA)  
- Support **real-time streaming** via Kafka  
- Enable **multi-agent collaboration**  
- Fine-tune **memo generation** using structured prompts  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a branch (`feature/your-feature`)  
3. Commit and push changes  
4. Open a Pull Request  

Keep functions modular, include tests, and follow clean code principles.

---

## 📜 License

**MIT License © 2025**  

Created and maintained as part of the **Smart Decision Support Agent Capstone Project**.

---

## 📞 Need Help?

1. Check the troubleshooting section above
2. Verify your `.env` file is in the project root
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check the error message - it usually tells you what's missing

---

## 💡 Pro Tips

1. **Start with Hugging Face** - It's free and works great for testing!
2. **Get a HF token** - Optional but gives higher rate limits
3. **Use smaller models for testing** - Faster responses
4. **Switch to OpenAI for production** - Better quality for final deployments
5. **Monitor API usage** - Check dashboards for costs (OpenAI) or rate limits (HF)
6. **Use `.env` file** - Keeps keys out of code
7. **Never commit `.env`** - Already in `.gitignore`
