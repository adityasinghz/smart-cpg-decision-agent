"""
Streamlit UI for the CPG Decision Support Agent.
Provides an interactive interface for querying the agent.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.agent_core import CPGDecisionAgent
from src.agent.memory import SessionMemory
from src.genai.llm_interface import LLMInterface
from src.data_loader import load_cpg_data, get_data_summary


def initialize_agent():
    """Initialize the agent with session state."""
    if 'agent' not in st.session_state:
        # Check which provider to use
        use_hf = st.session_state.get('use_huggingface', False)
        
        # If not explicitly set, default to Hugging Face if no OpenAI key
        if not use_hf:
            has_openai_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY'))
            if not has_openai_key:
                # No OpenAI key, default to Hugging Face
                use_hf = True
                st.session_state['use_huggingface'] = True
        
        if use_hf:
            # Use Hugging Face (no API key required!)
            try:
                llm = LLMInterface(
                    use_huggingface=True,
                    huggingface_model=st.session_state.get('hf_model', 'mistralai/Mistral-7B-Instruct-v0.2')
                )
            except Exception as e:
                st.error(f"⚠️ Error initializing Hugging Face: {e}")
                st.info("💡 Make sure you have installed: `pip install langchain-huggingface langchain-community`")
                st.stop()
        else:
            # Check for OpenAI/Azure API key
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
            if not api_key:
                st.error("⚠️ API key not found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
                st.info("💡 **Tip:** You can use Hugging Face for FREE! Enable it in the sidebar.")
                st.stop()
            
            # Initialize LLM
            use_azure = os.getenv('AZURE_OPENAI_ENDPOINT') is not None
            llm = LLMInterface(
                model=st.session_state.get('model', 'gpt-4'),
                use_azure=use_azure
            )
        
        # Initialize memory
        memory = SessionMemory()
        
        # Initialize agent
        data_path = st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
        if os.path.exists(data_path):
            st.session_state['agent'] = CPGDecisionAgent(
                llm=llm,
                memory=memory,
                data_path=data_path
            )
        else:
            st.session_state['agent'] = CPGDecisionAgent(
                llm=llm,
                memory=memory
            )
            st.warning(f"⚠️ Data file not found at {data_path}. Load data manually.")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="CPG Decision Support Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 CPG Decision Support Agent")
    st.markdown("Ask questions about your sales data and get AI-powered insights and recommendations.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if OpenAI key exists, default to HF if not
        has_openai_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY'))
        default_to_hf = not has_openai_key
        
        # Provider selection
        use_hf = st.checkbox("Use Hugging Face (FREE!)", value=st.session_state.get('use_huggingface', default_to_hf))
        st.session_state['use_huggingface'] = use_hf
        
        if use_hf:
            # Hugging Face model selection
            hf_model = st.selectbox(
                "Hugging Face Model",
                [
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "microsoft/phi-2",
                    "google/flan-t5-large",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "HuggingFaceH4/zephyr-7b-beta"
                ],
                index=0
            )
            st.session_state['hf_model'] = hf_model
            st.info("💡 Using free Hugging Face API. Get a token for higher limits: https://huggingface.co/settings/tokens")
        else:
            # OpenAI model selection
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-4", "gpt-3.5-turbo"],
                index=0
            )
            st.session_state['model'] = model
        
        # Data path
        data_path = st.text_input(
            "Data Path",
            value="data/cpg_sales_data.parquet"
        )
        st.session_state['data_path'] = data_path
        
        # Load data button
        if st.button("Load Data"):
            if os.path.exists(data_path):
                if 'agent' in st.session_state:
                    st.session_state['agent'].load_data(data_path)
                st.success("✅ Data loaded successfully!")
            else:
                st.error(f"❌ File not found: {data_path}")
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What are the sales trends for the last quarter?",
            "Compare performance across different stores",
            "What would happen if we run a 15% discount promotion?",
            "Detect any anomalies in sales data",
            "What are the seasonal patterns in our sales?",
            "Simulate a 10% price increase for product 101"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state['question'] = q
    
    # Initialize agent
    initialize_agent()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            value=st.session_state.get('question', ''),
            height=100,
            key='question_input'
        )
        
        # Generate memo checkbox
        generate_memo = st.checkbox("Generate Strategy Memo", value=True)
        
        # Submit button
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            if question:
                with st.spinner("Analyzing..."):
                    agent = st.session_state['agent']
                    result = agent.run(question, generate_memo=generate_memo)
                    
                    # Store result in session state
                    st.session_state['last_result'] = result
                    
                    # Display response
                    st.success("✅ Analysis complete!")
            else:
                st.warning("Please enter a question.")
        
        # Display results
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            st.divider()
            st.header("📋 Response")
            st.write(result['response'])
            
            if result.get('strategy_memo'):
                st.divider()
                st.header("📄 Strategy Memo")
                st.markdown(result['strategy_memo'])
            
            if result.get('analysis_results'):
                st.divider()
                st.header("📊 Analysis Results")
                
                for tool_name, tool_result in result['analysis_results'].items():
                    with st.expander(f"🔧 {tool_name}"):
                        st.json(tool_result)
            
            if result.get('tool_calls'):
                st.divider()
                st.caption(f"Tools used: {', '.join(result['tool_calls'])}")
    
    with col2:
        st.header("📈 Data Summary")
        
        if 'agent' in st.session_state and st.session_state['agent'].data is not None:
            try:
                summary = get_data_summary(st.session_state['agent'].data)
                
                st.metric("Total Rows", f"{summary.get('rows', 'N/A'):,}")
                st.metric("Stores", summary.get('stores', 'N/A'))
                st.metric("SKUs", summary.get('skus', 'N/A'))
                
                if summary.get('total_revenue'):
                    st.metric("Total Revenue", f"${summary['total_revenue']:,.2f}")
                
                if summary.get('date_range'):
                    st.caption(f"Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
            except Exception as e:
                st.error(f"Error loading summary: {e}")
        else:
            st.info("Load data to see summary")
        
        st.divider()
        
        # Memory info
        if 'agent' in st.session_state:
            memory = st.session_state['agent'].memory
            st.header("🧠 Memory")
            st.caption(f"Conversation turns: {len(memory.conversation_history)}")
            st.caption(f"Tool calls: {len(memory.tool_calls)}")
            
            if st.button("Clear Memory"):
                memory.clear()
                st.success("Memory cleared!")
                st.rerun()


if __name__ == "__main__":
    main()
