# Smart CPG Decision Agent - Simple Explanation (For 17-Year-Olds)

## What is this project?
An app that helps businesses understand their sales data. Instead of complex spreadsheets, you ask questions in plain English and get answers.

---

## Main Files and How They Work

### 1. Starting the App
**File:** `src/ui/streamlit_app.py`
- **Function:** `main()` - This starts everything
- **Function:** `init_session_state()` - Sets up memory
- **Function:** `render_sidebar()` - Shows navigation buttons

When you run `streamlit run src/ui/streamlit_app.py`, it calls `main()` and opens the web interface.

---

### 2. Loading Your Data
**File:** `src/data_loader.py`
- **Function:** `load_cpg_data()` - Reads your CSV/Parquet file
- **Function:** `get_data_summary()` - Calculates basic stats

**File:** `src/ui/streamlit_app.py`
- **Function:** `render_home()` - Shows the Overview page
- When you click "Load Data", it calls `load_cpg_data()` and stores data in `st.session_state.data`

---

### 3. The AI Brain (Agent)
**File:** `src/agent/agent_core.py`
- **Class:** `CPGDecisionAgent` - This is the main brain
- **Function:** `chat()` - Handles your questions
- **Function:** `_plan_actions()` - Decides what to do
- **Function:** `_execute_plan()` - Runs the tools
- **Function:** `_generate_response()` - Creates the answer

**How it works:**
1. You ask: "What are sales trends?"
2. `chat()` receives your question
3. `_plan_actions()` classifies it as "trend analysis"
4. `_execute_plan()` calls the trend tool
5. `_generate_response()` uses AI to write the answer

---

### 4. The Tools (Analyzers)

**File:** `src/tools/trend_analysis.py`
- **Function:** `extract_trends()` - Finds if sales are going up/down
- **Function:** `calculate_growth_rate()` - Calculates growth percentage

**File:** `src/tools/anomaly_detection.py`
- **Function:** `get_anomaly_summary()` - Finds weird numbers
- Uses 3 methods: IQR, Z-score, Isolation Forest

**File:** `src/tools/scenario_simulation.py`
- **Function:** `simulate_price_change()` - "What if we change price?"
- **Function:** `simulate_promotion()` - "What if we run a sale?"

---

### 5. The AI Connection
**File:** `src/genai/llm_interface.py`
- **Class:** `LLMInterface` - Talks to AI models
- Connects to OpenAI, Azure, or HuggingFace
- The agent uses this to generate answers

---

### 6. Memory System
**File:** `src/agent/memory.py`
- **Class:** `SessionMemory` - Remembers things
- **Function:** `add_message()` - Saves conversations
- **Function:** `update_context()` - Stores data info

This lets the app remember what you asked before!

---

## Complete Flow with File Names

### When You Start the App:
```
1. You run: streamlit run src/ui/streamlit_app.py
2. streamlit_app.py → main() function starts
3. main() → calls init_session_state()
4. main() → calls render_sidebar()
5. main() → shows render_home() (Overview page)
```

### When You Load Data:
```
1. User clicks "Load Data" button
2. render_home() → calls load_cpg_data() from data_loader.py
3. load_cpg_data() → reads your file → returns DataFrame
4. render_home() → stores in st.session_state.data
5. render_home() → calls initialize_agent()
6. initialize_agent() → creates CPGDecisionAgent
7. Agent → calls load_data() → profiles the data
8. Agent → stores profile in SessionMemory
```

### When You Ask a Question (AI Assistant):
```
1. User types question in render_chat()
2. render_chat() → calls agent.chat(question)
3. agent_core.py → chat() function:
   ├─→ add_message('user', question) to memory
   ├─→ _plan_actions(question) → decides what tool to use
   ├─→ _execute_plan(plan) → calls the tool:
   │   ├─→ If trend: calls extract_trends() from trend_analysis.py
   │   ├─→ If anomaly: calls get_anomaly_summary() from anomaly_detection.py
   │   └─→ If scenario: calls simulate_price_change() from scenario_simulation.py
   ├─→ _generate_response() → uses LLMInterface to create answer
   └─→ add_message('assistant', response) to memory
4. render_chat() → displays the answer
```

### When You Use a Module (Like Anomaly Detection):
```
1. User clicks "Anomaly Detection" in sidebar
2. main() → calls render_anomaly_detection()
3. render_anomaly_detection() → user selects metric and method
4. User clicks "Detect" button
5. render_anomaly_detection() → calls get_anomaly_summary() from anomaly_detection.py
6. get_anomaly_summary() → runs 3 detection methods
7. Results displayed with charts (using Plotly)
```

---

## All the Pages and Their Functions

**File:** `src/ui/streamlit_app.py`

1. **Overview Page**
   - Function: `render_home()`
   - What it does: Shows data loading, quick stats

2. **AI Assistant Page**
   - Function: `render_chat()`
   - What it does: Chat interface, calls `agent.chat()`

3. **Business Insights Page**
   - Function: `render_smart_insights()`
   - What it does: Auto-generates insights using agent

4. **Anomaly Detection Page**
   - Function: `render_anomaly_detection()`
   - What it does: Calls `get_anomaly_summary()` from `anomaly_detection.py`

5. **Data Comparison Page**
   - Function: `render_comparison_tool()`
   - What it does: Compares data segments

6. **Forecasting Page**
   - Function: `render_forecasting()`
   - What it does: Predicts future sales

7. **Custom Reports Page**
   - Function: `render_custom_reports()`
   - What it does: Creates custom reports

8. **Alert Management Page**
   - Function: `render_alert_system()`
   - What it does: Sets up alerts

9. **Dashboard Page**
   - Function: `render_performance_dashboard()`
   - What it does: Shows real-time KPIs

---

## File Structure Summary

```
smart-cpg-decision-agent/
├── src/
│   ├── ui/
│   │   └── streamlit_app.py          ← Main web interface (all pages)
│   ├── agent/
│   │   ├── agent_core.py            ← The AI brain (CPGDecisionAgent)
│   │   └── memory.py                ← Remembers conversations
│   ├── tools/
│   │   ├── trend_analysis.py        ← Analyzes trends
│   │   ├── anomaly_detection.py     ← Finds anomalies
│   │   └── scenario_simulation.py    ← "What-if" scenarios
│   ├── genai/
│   │   └── llm_interface.py         ← Talks to AI models
│   └── data_loader.py               ← Loads your data
└── data/
    └── cpg_sales_data.parquet        ← Your sales data file
```

---

## Real Example with File Names

**You ask:** "What are the sales trends?"

**What happens:**
1. `streamlit_app.py` → `render_chat()` receives your question
2. `render_chat()` → calls `agent.chat("What are the sales trends?")`
3. `agent_core.py` → `chat()` function:
   - Calls `_plan_actions()` → decides it's a "trend" question
   - Calls `_execute_plan()` → runs `trend_analysis` tool
4. `trend_analysis.py` → `extract_trends()` analyzes the data
5. Returns: `{'direction': 'increasing', 'growth_rate': 12.5%}`
6. `agent_core.py` → `_generate_response()` uses `LLMInterface`
7. `llm_interface.py` → sends to AI model → gets answer
8. `render_chat()` → displays the answer to you

---

## Key Functions to Remember

**In `agent_core.py`:**
- `chat()` - Main function that handles questions
- `_plan_actions()` - Decides what to do
- `_execute_plan()` - Runs the tools
- `_generate_response()` - Creates the answer

**In `streamlit_app.py`:**
- `main()` - Starts the app
- `render_home()` - Overview page
- `render_chat()` - AI Assistant page
- `render_anomaly_detection()` - Anomaly Detection page
- (and 6 more render functions for other pages)

**In tools:**
- `extract_trends()` - Finds trends
- `get_anomaly_summary()` - Finds anomalies
- `simulate_price_change()` - Simulates scenarios

---

## Bottom Line

- `streamlit_app.py` = The interface (buttons, pages)
- `agent_core.py` = The brain (understands questions, coordinates tools)
- `tools/` = The analyzers (do the math)
- `llm_interface.py` = Talks to AI (generates answers)
- `memory.py` = Remembers things (conversation history)

When you ask a question, `agent_core.py` coordinates everything, calls the right tool, and uses `llm_interface.py` to give you an answer.

---

## Simple Analogy

Think of it like a restaurant:
- **`streamlit_app.py`** = The menu and waiter (what you see and interact with)
- **`agent_core.py`** = The head chef (coordinates everything)
- **`tools/`** = The kitchen staff (do the actual work)
- **`llm_interface.py`** = The recipe book (knows how to make things)
- **`memory.py`** = The order pad (remembers what you asked for)

You order food (ask a question) → Waiter takes order → Head chef decides what to cook → Kitchen staff prepares it → You get your meal (answer)!

---

## How to Run It

1. **Start the app:**
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

2. **Load data:**
   - Click "Load Data" button on Overview page
   - Uses `load_cpg_data()` from `data_loader.py`

3. **Ask questions:**
   - Go to AI Assistant page
   - Type your question
   - `render_chat()` → `agent.chat()` → Tools → Answer!

---

## What Makes It Special

✅ **Smart**: Automatically picks the right tool for your question  
✅ **Remembers**: Knows what you asked before  
✅ **Fast**: Answers in seconds  
✅ **Easy**: Just ask in plain English  
✅ **Accurate**: Checks everything against real data  

---

**Created for easy understanding - Share this with anyone who wants to know how the project works!**

