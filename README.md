# Smart CPG Decision Support Agent

## Overview
The **Smart CPG Decision Support Agent** uses **Generative AI (GenAI)** and **Agentic AI** to help Consumer Packaged Goods (CPG) businesses analyze sales data and make data-driven decisions.

It ingests multi-store, multi-SKU sales data, detects trends and anomalies, simulates business scenarios, and produces natural-language strategy memos through an agentic reasoning loop.

---

## 🚀 Features
- **Automated Data Ingestion:** Load and preprocess large, complex datasets using PySpark or pandas.  
- **Trend & Anomaly Detection:** Identify seasonality, promotions, and unexpected shifts in sales.  
- **Scenario Simulation:** Run “what-if” simulations such as price hikes or promo campaigns.  
- **AI-Generated Insights:** Summarize and explain outcomes in natural language.  
- **Agentic Loop:** Use LangChain or CrewAI to choose and execute analysis tools automatically.  
- **Interactive UI:** Streamlit dashboard or CLI interface for conversational insights.  

---

## 🧱 Project Structure
```
smart-cpg-decision-agent/
├── data/
│   └── cpg_sales_data.parquet
│
├── notebooks/
│   ├── 01_EDA_and_Data_Loading.ipynb
│   ├── 02_Trend_Anomaly_Detection.ipynb
│   ├── 03_Scenario_Simulation.ipynb
│   └── 04_Agent_Loop_Prototype.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── tools/
│   │   ├── trend_analysis.py
│   │   ├── anomaly_detection.py
│   │   └── scenario_simulation.py
│   ├── genai/
│   │   └── llm_interface.py
│   ├── agent/
│   │   ├── agent_core.py
│   │   └── memory.py
│   └── ui/
│       ├── streamlit_app.py
│       └── cli.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_tools.py
│   └── test_agent.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/smart-cpg-decision-agent.git
cd smart-cpg-decision-agent
```

### 2. Create and Activate a Virtual Environment
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

### 4. Run Tests
```bash
pytest -q
```

---

## ☁️ Databricks Integration

1. Upload your dataset (`.csv` or `.parquet`) to **Azure Databricks**.  
2. Connect this repository to **Databricks Repos**.  
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

## 🧠 Agentic Architecture

| Layer | Purpose |
|-------|----------|
| **Data Layer** | Load, clean, and validate historical sales data |
| **Tool Layer** | Trend, anomaly, and simulation utilities |
| **GenAI Layer** | Use OpenAI / Azure OpenAI for generating insights |
| **Agent Layer** | Manage tool orchestration and context (LangChain / CrewAI) |
| **UI Layer** | Streamlit and CLI interfaces for interaction |

**Flow Example:**
1. User asks, “What if we apply a 10% discount on Beverages in the North region?”  
2. Agent picks the right simulation tools.  
3. Simulation produces new KPIs.  
4. LLM generates a summary memo with recommendations.  
5. UI displays graphs and text insights.

---

## 💡 Synthetic Dataset Generator

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

## 🔐 Environment Variables
Create a `.env` file (never commit it):
```
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
DATA_PATH=data/cpg_sales_data.parquet
```

---

## 🧰 .gitignore
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

## 🧩 requirements.txt
```
pandas>=1.5
pyspark>=3.3
streamlit>=1.20
langchain>=0.1
openai>=0.27
scikit-learn>=1.1
matplotlib>=3.5
pytest>=7.0
python-dotenv>=0.21
```

---

## 🖥️ Running Components

### Streamlit UI
```bash
streamlit run src/ui/streamlit_app.py
```

### CLI Interface
```bash
python src/ui/cli.py
```

### Run Tests
```bash
pytest tests/
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
