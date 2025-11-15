"""
Enhanced Streamlit UI for the CPG Decision Support Agent.
Modern, multi-page interface with interactive visualizations and improved UX.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.agent_core import CPGDecisionAgent
from src.agent.memory import SessionMemory
from src.genai.llm_interface import LLMInterface
from src.data_loader import load_cpg_data, get_data_summary
from src.tools.trend_analysis import extract_trends, calculate_growth_rate
from src.tools.anomaly_detection import detect_anomalies, get_anomaly_summary
from src.tools.scenario_simulation import simulate_promotion, simulate_price_change

# -----------------------
# Page Configuration
# -----------------------

st.set_page_config(
    page_title="CPG Decision Support Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS for Modern UI
# -----------------------

# Load CSS from external file
css_file_path = Path(__file__).parent / "styles.css"
if css_file_path.exists():
    with open(css_file_path, 'r', encoding='utf-8') as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ CSS file not found. Using default styles.")

# -----------------------
# Session State Initialization
# -----------------------

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'agent': None,
        'data_loaded': False,
        'data': None,
        'metadata': None,
        'chat_history': [],
        'last_result': None,
        'use_huggingface': False,
        'hf_model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'model': 'gpt-4',
        'data_path': 'data/cpg_sales_data.parquet',
        'analysis_results': {},
        'chart_height': 450
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------
# Agent Initialization
# -----------------------

def initialize_agent():
    """Initialize the agent with caching."""
    use_hf = st.session_state.get('use_huggingface', False)
    
    # Auto-detect if no OpenAI key
    if not use_hf:
        has_openai_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY'))
        if not has_openai_key:
            use_hf = True
            st.session_state['use_huggingface'] = True
    
    try:
        if use_hf:
            llm = LLMInterface(
                use_huggingface=True,
                huggingface_model=st.session_state.get('hf_model', 'mistralai/Mistral-7B-Instruct-v0.2')
            )
        else:
            use_azure = os.getenv('AZURE_OPENAI_ENDPOINT') is not None
            llm = LLMInterface(
                model=st.session_state.get('model', 'gpt-4'),
                use_azure=use_azure
            )
        
        memory = SessionMemory()
        data_path = st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
        
        if os.path.exists(data_path):
            agent = CPGDecisionAgent(llm=llm, memory=memory, data_path=data_path)
            return agent, True
        else:
            agent = CPGDecisionAgent(llm=llm, memory=memory)
            return agent, False
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None, False

# -----------------------
# Sidebar
# -----------------------

def render_sidebar():
    """Render sidebar with navigation and configuration."""
    with st.sidebar:
        st.markdown("# 📊 CPG Agent")
        st.markdown("---")
        
        # Navigation with beautiful buttons
        st.markdown("## 🎯 Navigation")
        
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "🏠 Home"
        
        # Navigation pages
        pages = {
            "🏠 Home": "🏠 Home",
            "💬 AI Chat": "💬 AI Chat",
            "📈 Analytics": "📈 Analytics",
            "📊 Dashboard": "📊 Dashboard",
            "🕘 Chat History": "🕘 Chat History",
            "⚙️ Settings": "⚙️ Settings"
        }
        
        # Create navigation buttons
        for page_name, page_value in pages.items():
            is_active = st.session_state.current_page == page_value
            
            if st.button(
                page_name,
                key=f"nav_{page_value}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page_value
                st.rerun()
        
        st.markdown("---")
        
        # Data Status
        if st.session_state.data_loaded and st.session_state.metadata:
            st.markdown("### 📦 Data Status")
            st.success("✓ Data Loaded")
            md = st.session_state.metadata
            st.metric("Total Records", f"{md.get('rows', 0):,}")
            if md.get('date_range'):
                d0, d1 = md['date_range']
                st.caption(f"📅 {pd.to_datetime(d0).strftime('%Y-%m-%d')} to {pd.to_datetime(d1).strftime('%Y-%m-%d')}")
            st.metric("Stores", md.get('stores', 'N/A'))
            st.metric("SKUs", md.get('skus', 'N/A'))
            if md.get('total_revenue'):
                st.metric("Total Revenue", f"${md['total_revenue']:,.0f}")
        else:
            st.warning("⚠️ Data not loaded")
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            # Provider selection
            has_openai_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY'))
            default_to_hf = not has_openai_key
            
            use_hf = st.checkbox(
                "Use Hugging Face (FREE!)",
                value=st.session_state.get('use_huggingface', default_to_hf)
            )
            st.session_state['use_huggingface'] = use_hf
            
            if use_hf:
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
                st.info("💡 Free API. Get token for higher limits: https://huggingface.co/settings/tokens")
            else:
                model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4", "gpt-3.5-turbo"],
                    index=0
                )
                st.session_state['model'] = model
            
            # Data path
            data_path = st.text_input(
                "Data Path",
                value=st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
            )
            st.session_state['data_path'] = data_path
            
            # Chart height
            chart_height = st.slider("Chart Height", 300, 800, st.session_state.chart_height)
            st.session_state.chart_height = chart_height
        
        st.markdown("---")
        
        # Memory info
        if st.session_state.agent:
            st.markdown("### 🧠 Memory")
            memory = st.session_state.agent.memory
            st.caption(f"Conversations: {len(memory.conversation_history)}")
            st.caption(f"Tool calls: {len(memory.tool_calls)}")
            
            if st.button("🗑️ Clear Memory", use_container_width=True):
                memory.clear()
                st.success("Memory cleared!")
                st.rerun()

# -----------------------
# Home Page
# -----------------------

def render_home():
    """Render modern interactive home page with enhanced styling."""
    
    # Hero Section
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">📊 CPG Decision Support Agent</h1>
            <p class="hero-subtitle">AI-Powered Analytics for Consumer Packaged Goods</p>
            <p class="hero-description">
                Transform your sales data into actionable insights with advanced AI analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Highlights
        st.markdown("### ✨ Key Features")
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        features = [
            ("🤖", "AI Chat", "Ask questions in natural language and get instant AI-powered insights"),
            ("📈", "Analytics", "Advanced trend analysis, anomaly detection, and scenario simulations"),
            ("📊", "Dashboard", "Interactive visualizations and comprehensive business metrics"),
            ("🎯", "Insights", "Real-time data analysis with actionable recommendations")
        ]
        
        for i, (icon, title, desc) in enumerate(features):
            with [feature_col1, feature_col2, feature_col3, feature_col4][i]:
                st.markdown(f"""
                <div class="feature-card">
                    <span class="feature-icon">{icon}</span>
                    <h3 class="feature-title">{title}</h3>
                    <p class="feature-text">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Load Data Section
        st.markdown("### 🚀 Get Started")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 25px; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea; margin-bottom: 1rem; font-size: 1.8rem;">Ready to Explore Your Data?</h3>
                <p style="color: #4a5568; margin-bottom: 2rem; font-size: 1.1rem;">Load your sales data and unlock powerful AI-driven insights</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Load Data & Initialize Agent", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading data and initializing AI agent..."):
                    try:
                        data_path = st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
                        if os.path.exists(data_path):
                            data = load_cpg_data(data_path)
                            metadata = get_data_summary(data)
                            
                            st.session_state.data = data
                            st.session_state.metadata = metadata
                            st.session_state.data_loaded = True
                            
                            # Initialize agent
                            agent, data_loaded = initialize_agent()
                            if agent:
                                st.session_state.agent = agent
                                if not data_loaded:
                                    agent.load_data(data_path)
                            
                            st.success("✅ System initialized successfully!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ Data file not found: {data_path}")
                            st.info("💡 Please check the file path in Settings or ensure the data file exists.")
                    except Exception as e:
                        st.error(f"❌ Error loading data: {e}")
        
        # Data File Info with Enhanced Styling
        data_path = st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
        file_col1, file_col2 = st.columns([2, 1])
        
        with file_col1:
            if os.path.exists(data_path):
                file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
                st.markdown(f"""
                <div class="file-status" style="border-left-color: #48bb78; background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);">
                    <strong style="color: #22543d;">✅ Data File Ready</strong><br>
                    <span style="color: #2d5016;">📁 {data_path}</span><br>
                    <span style="color: #22543d; font-size: 0.9rem;">📊 Size: {file_size:.2f} MB</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="file-status" style="border-left-color: #f56565; background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);">
                    <strong style="color: #742a2a;">⚠️ Data File Not Found</strong><br>
                    <span style="color: #742a2a;">📁 {data_path}</span><br>
                    <span style="color: #742a2a; font-size: 0.9rem;">💡 Please check Settings to update the path</span>
                </div>
                """, unsafe_allow_html=True)
        
        with file_col2:
            # Align button vertically with file status card
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if st.button("⚙️ Go to Settings", use_container_width=True, type="secondary", key="settings_btn_home"):
                st.session_state.current_page = "⚙️ Settings"
                st.rerun()
        
        st.markdown("---")
        
        # Enhanced Quick Guide with Interactive Cards
        st.markdown("### 📖 Quick Guide")
        guide_col1, guide_col2 = st.columns(2)
        
        with guide_col1:
            st.markdown("""
            <div class="guide-card">
                <h4 style="color: #667eea; margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎯</span>
                    What You Can Do
                </h4>
                <ul style="color: #4a5568; line-height: 2.2; list-style: none; padding-left: 0;">
                    <li style="margin: 0.75rem 0;">✨ Ask natural language questions about your sales data</li>
                    <li style="margin: 0.75rem 0;">📈 Analyze trends and detect anomalies</li>
                    <li style="margin: 0.75rem 0;">🎯 Simulate business scenarios (promotions, price changes)</li>
                    <li style="margin: 0.75rem 0;">📊 Explore interactive dashboards</li>
                    <li style="margin: 0.75rem 0;">🤖 Get AI-powered recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with guide_col2:
            st.markdown("""
            <div class="guide-card">
                <h4 style="color: #667eea; margin-bottom: 1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">💡</span>
                    Example Questions
                </h4>
                <ul style="color: #4a5568; line-height: 2.2; list-style: none; padding-left: 0;">
                    <li style="margin: 0.75rem 0;">💬 "What are the sales trends?"</li>
                    <li style="margin: 0.75rem 0;">📊 "Compare store performance"</li>
                    <li style="margin: 0.75rem 0;">🎁 "Simulate a 20% discount promotion"</li>
                    <li style="margin: 0.75rem 0;">🔍 "Detect anomalies in revenue"</li>
                    <li style="margin: 0.75rem 0;">🏆 "What are the top categories?"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature Stats (Before Loading)
        st.markdown("### 🌟 Platform Features")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        stats = [
            ("AI Models", "Multiple LLM Support"),
            ("Real-time", "Live Analytics"),
            ("Interactive", "Dynamic Visualizations"),
            ("Smart", "Auto Insights")
        ]
        
        for i, (title, desc) in enumerate(stats):
            with [stats_col1, stats_col2, stats_col3, stats_col4][i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 15px; border: 2px solid #e2e8f0; transition: all 0.3s ease;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{"🤖" if i == 0 else "⚡" if i == 1 else "📊" if i == 2 else "🧠"}</div>
                    <div style="font-weight: 700; color: #667eea; margin-bottom: 0.25rem;">{title}</div>
                    <div style="font-size: 0.875rem; color: #718096;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Data Loaded - Enhanced Dashboard View
    if st.session_state.data_loaded and st.session_state.metadata:
        md = st.session_state.metadata
        data = st.session_state.data
        
        # Welcome Banner
        st.markdown("""
        <div class="welcome-banner">
            <h2 class="welcome-title">🎉 Welcome Back!</h2>
            <p class="welcome-text">Your data is loaded and ready for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Metrics Cards
        st.markdown("### 📊 Overview Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        metrics_data = [
            ("Total Revenue", f"${md.get('total_revenue', 0):,.0f}", "#667eea"),
            ("Avg Transaction", f"${(md.get('total_revenue', 0.0) / max(md.get('rows', 1), 1)):.2f}", "#48bb78"),
            ("Total Stores", str(md.get('stores', 'N/A')), "#f5576c"),
            ("Total SKUs", str(md.get('skus', 'N/A')), "#fbbf24"),
            ("Date Range", f"{(data['date'].max() - data['date'].min()).days} days" if 'date' in data.columns else "N/A", "#764ba2")
        ]
        
        for i, (label, value, color) in enumerate(metrics_data):
            with [metric_col1, metric_col2, metric_col3, metric_col4, metric_col5][i]:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color};">
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        actions = [
            ("💬", "Go to AI Chat", "💬 AI Chat"),
            ("📈", "View Analytics", "📈 Analytics"),
            ("📊", "Open Dashboard", "📊 Dashboard"),
            ("🔄", "Reload Data", None)
        ]
        
        for i, (icon, label, page) in enumerate(actions):
            with [action_col1, action_col2, action_col3, action_col4][i]:
                if st.button(f"{icon} {label}", key=f"action_{i}", use_container_width=True, type="primary" if i == 0 else "secondary"):
                    if page:
                        st.session_state.current_page = page
                        st.rerun()
                    else:
                        st.session_state.data_loaded = False
                        st.session_state.data = None
                        st.session_state.metadata = None
                        st.session_state.agent = None
                        st.success("Data cleared. Click 'Load Data' to reload.")
                        st.rerun()
        
        st.markdown("---")
        
        # Enhanced Quick Insights with Visualizations
        st.markdown("### 🎯 Quick Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            <div class="insight-card">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Top Categories by Revenue</h3>
            </div>
            """, unsafe_allow_html=True)
            if 'category' in data.columns and 'revenue' in data.columns:
                cat_rev = data.groupby('category')['revenue'].sum().sort_values(ascending=False).head(5)
                
                # Create interactive bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=cat_rev.values,
                    y=cat_rev.index,
                    orientation='h',
                    marker=dict(
                        color=cat_rev.values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Revenue")
                    ),
                    text=[f"${x:,.0f}" for x in cat_rev.values],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>'
                ))
                fig.update_layout(
                    title='',
                    xaxis_title='Revenue ($)',
                    yaxis_title='Category',
                    height=300,
                    template='plotly_white',
                    showlegend=False,
                    margin=dict(l=20, r=20, t=10, b=20)
                )
                config = {'displayModeBar': False}
                st.plotly_chart(fig, use_container_width=True, config=config)
        
        with insight_col2:
            st.markdown("""
            <div class="insight-card">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🗺️ Top Regions by Revenue</h3>
            </div>
            """, unsafe_allow_html=True)
            if 'store_region' in data.columns and 'revenue' in data.columns:
                region_rev = data.groupby('store_region')['revenue'].sum().sort_values(ascending=False)
                
                # Create interactive pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=region_rev.index,
                    values=region_rev.values,
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3),
                    hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
                )])
                fig.update_layout(
                    title='',
                    height=300,
                    template='plotly_white',
                    annotations=[dict(text='Regions', x=0.5, y=0.5, font_size=14, showarrow=False)],
                    margin=dict(l=20, r=20, t=10, b=20)
                )
                config = {'displayModeBar': False}
                st.plotly_chart(fig, use_container_width=True, config=config)
        
        st.markdown("---")
        
        # Data Summary Cards
        st.markdown("### 📋 Data Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            if 'date' in data.columns:
                min_date = data['date'].min().strftime('%Y-%m-%d')
                max_date = data['date'].max().strftime('%Y-%m-%d')
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #667eea;">
                    <strong style="color: #667eea; font-size: 1rem;">📅 Date Range</strong><br>
                    <span style="color: #2d3748; font-size: 0.95rem; font-weight: 500;">{min_date} to {max_date}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with summary_col2:
            if 'units_sold' in data.columns:
                total_units = data['units_sold'].sum()
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #48bb78;">
                    <strong style="color: #48bb78; font-size: 1rem;">📦 Total Units</strong><br>
                    <span style="color: #2d3748; font-size: 0.95rem; font-weight: 500;">{total_units:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with summary_col3:
            if 'price' in data.columns and 'revenue' in data.columns and 'units_sold' in data.columns:
                avg_price = data['revenue'].sum() / data['units_sold'].sum() if data['units_sold'].sum() > 0 else 0
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #f5576c;">
                    <strong style="color: #f5576c; font-size: 1rem;">💰 Avg Price</strong><br>
                    <span style="color: #2d3748; font-size: 0.95rem; font-weight: 500;">${avg_price:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with summary_col4:
            if 'promo_flag' in data.columns:
                promo_rate = data['promo_flag'].mean() * 100
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #fbbf24;">
                    <strong style="color: #fbbf24; font-size: 1rem;">🎁 Promo Rate</strong><br>
                    <span style="color: #2d3748; font-size: 0.95rem; font-weight: 500;">{promo_rate:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Getting Started Guide with Tabs
        st.markdown("### 🚀 Getting Started")
        
        guide_tab1, guide_tab2, guide_tab3 = st.tabs(["💬 AI Chat", "📈 Analytics", "📊 Dashboard"])
        
        with guide_tab1:
            st.markdown("""
            <div style="padding: 1.5rem; background: #f7fafc; border-radius: 15px; margin: 1rem 0;">
                <h4 style="color: #667eea; margin-bottom: 1rem;">Ask questions in natural language:</h4>
                <ul style="color: #4a5568; line-height: 2;">
                    <li>"What are the sales trends?"</li>
                    <li>"Compare performance across stores"</li>
                    <li>"What would happen with a 15% discount?"</li>
                    <li>"Detect anomalies in sales data"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to AI Chat →", key="guide_chat", use_container_width=True):
                st.session_state.current_page = "💬 AI Chat"
                st.rerun()
        
        with guide_tab2:
            st.markdown("""
            <div style="padding: 1.5rem; background: #f7fafc; border-radius: 15px; margin: 1rem 0;">
                <h4 style="color: #667eea; margin-bottom: 1rem;">Explore advanced analytics:</h4>
                <ul style="color: #4a5568; line-height: 2;">
                    <li>Trend analysis with forecasting</li>
                    <li>Anomaly detection</li>
                    <li>Scenario simulations</li>
                    <li>Seasonal pattern analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Analytics →", key="guide_analytics", use_container_width=True):
                st.session_state.current_page = "📈 Analytics"
                st.rerun()
        
        with guide_tab3:
            st.markdown("""
            <div style="padding: 1.5rem; background: #f7fafc; border-radius: 15px; margin: 1rem 0;">
                <h4 style="color: #667eea; margin-bottom: 1rem;">View comprehensive dashboards:</h4>
                <ul style="color: #4a5568; line-height: 2;">
                    <li>Interactive visualizations</li>
                    <li>Real-time filtering</li>
                    <li>Performance comparisons</li>
                    <li>Executive KPIs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Dashboard →", key="guide_dashboard", use_container_width=True):
                st.session_state.current_page = "📊 Dashboard"
                st.rerun()
        
        st.markdown("---")
        
        # Example Questions with Interactive Cards
        st.markdown("### 💡 Try These Questions")
        example_questions = [
            ("What are the sales trends for the last quarter?", "trends", "📈"),
            ("Compare performance across different stores", "compare", "🏪"),
            ("What would happen if we run a 15% discount promotion?", "promo", "🎁"),
            ("Detect any anomalies in sales data", "anomalies", "🔍"),
            ("What are the seasonal patterns in our sales?", "seasonal", "🌊"),
            ("Simulate a 10% price increase for product 101", "simulate", "💰")
        ]
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            for question, key, icon in example_questions[:3]:
                st.markdown(f"""
                <div class="question-card" onclick="this.style.transform='scale(0.98)'; setTimeout(() => this.style.transform='', 200);">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.5rem;">{icon}</span>
                        <span style="color: #2d3748; font-weight: 500;">{question}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Ask →", key=f"example_{key}", use_container_width=True, type="secondary"):
                    st.session_state.current_page = "💬 AI Chat"
                    st.session_state['question'] = question
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)
        
        with example_col2:
            for question, key, icon in example_questions[3:]:
                st.markdown(f"""
                <div class="question-card" onclick="this.style.transform='scale(0.98)'; setTimeout(() => this.style.transform='', 200);">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.5rem;">{icon}</span>
                        <span style="color: #2d3748; font-weight: 500;">{question}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Ask →", key=f"example_{key}_2", use_container_width=True, type="secondary"):
                    st.session_state.current_page = "💬 AI Chat"
                    st.session_state['question'] = question
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status & Quick Stats
        st.markdown("### ⚡ System Status")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            agent_status = "✅ Active" if st.session_state.agent else "⚠️ Not Initialized"
            status_color = "#48bb78" if st.session_state.agent else "#f56565"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid {status_color};">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🤖</div>
                <div style="font-weight: 600; color: {status_color};">Agent</div>
                <div style="font-size: 0.875rem; color: #718096; margin-top: 0.25rem;">{agent_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col2:
            data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
            status_color = "#48bb78" if st.session_state.data_loaded else "#f56565"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid {status_color};">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: {status_color};">Data</div>
                <div style="font-size: 0.875rem; color: #718096; margin-top: 0.25rem;">{data_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col3:
            chat_count = len(st.session_state.chat_history)
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid #667eea;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💬</div>
                <div style="font-weight: 600; color: #667eea;">Messages</div>
                <div style="font-size: 0.875rem; color: #718096; margin-top: 0.25rem;">{chat_count} total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col4:
            analysis_count = len(st.session_state.analysis_results)
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 2px solid #fbbf24;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📈</div>
                <div style="font-weight: 600; color: #fbbf24;">Analyses</div>
                <div style="font-size: 0.875rem; color: #718096; margin-top: 0.25rem;">{analysis_count} completed</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------
# AI Chat Page
# -----------------------

def render_chat():
    """Render AI chat interface."""
    st.markdown("## 💬 AI-Powered Chat Assistant")
    st.caption("Ask questions in natural language about your sales data.")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    # Initialize agent if needed
    if not st.session_state.agent:
        agent, _ = initialize_agent()
        if agent:
            st.session_state.agent = agent
    
    # Force reinitialization if agent doesn't have chat method (for cached agents)
    if st.session_state.agent and not hasattr(st.session_state.agent, 'chat'):
        st.info("🔄 Updating agent... Please refresh the page or restart Streamlit.")
        agent, _ = initialize_agent()
        if agent:
            st.session_state.agent = agent
    
    if not st.session_state.agent:
        st.error("❌ Agent not initialized. Please check settings.")
        return
    
    # Chat history display
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg['content'])
                if 'analysis_results' in msg:
                    with st.expander("📊 Analysis Results"):
                        for tool_name, result in msg['analysis_results'].items():
                            st.json(result)
    
    # Chat input
    question = st.chat_input("Ask a question... (e.g., 'What is the sales trend?')")
    
    if question or st.session_state.get('question'):
        if question:
            q = question
        else:
            q = st.session_state.get('question', '')
            st.session_state['question'] = ''
        
        st.session_state.chat_history.append({'role': 'user', 'content': q})
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Analyzing..."):
                try:
                    # Use fast chat() method if available, otherwise fall back to run()
                    if hasattr(st.session_state.agent, 'chat'):
                        response = st.session_state.agent.chat(q)
                    else:
                        # Fallback for cached agents that don't have chat() yet
                        result = st.session_state.agent.run(q, generate_memo=False)
                        response = result.get('response', 'No response generated.')
                    
                    st.write(response)
                    
                    # Store in chat history
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    
                    # Optionally get detailed results for display (non-blocking)
                    # This can be done asynchronously or on-demand
                    if hasattr(st.session_state.agent, 'memory') and st.session_state.agent.memory.tool_calls:
                        recent_tool_calls = st.session_state.agent.memory.tool_calls[-3:]  # Last 3 tool calls
                        if recent_tool_calls:
                            with st.expander("📊 Analysis Results"):
                                for tc in recent_tool_calls:
                                    st.markdown(f"**{tc.get('tool_name', 'Unknown')}**")
                                    st.json(tc.get('result', {}))
                    
                except Exception as e:
                    error_msg = f"I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
        
        st.rerun()
    
    # Quick actions (fast - bypass LLM, call tools directly)
    st.markdown("---")
    st.markdown("#### 🎯 Quick Actions")
    
    def qa_show_trends():
        """Fast trend analysis without LLM."""
        try:
            from src.tools.trend_analysis import extract_trends, calculate_growth_rate
            result = extract_trends(st.session_state.data, date_col='date', value_col='revenue')
            st.session_state.analysis_results["trend"] = result
            
            trend_dir = result.get('trend_direction', 'unknown')
            slope = result.get('slope', 0)
            r2 = result.get('r_squared', 0)
            strength = result.get('trend_strength', 'unknown')
            
            reply = (
                "Here are the sales trends:\n"
                f"- Trend: {trend_dir.title()}, slope={slope:.4f}, R²={r2:.3f}\n"
                f"- Strength: {strength.title()}\n"
                f"- Intercept: {result.get('intercept', 0):.2f}"
            )
            st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            st.session_state.chat_history.append({'role': 'assistant', 'content': f"Error computing trends: {e}"})
    
    def qa_find_anomalies():
        """Fast anomaly detection without LLM."""
        try:
            from src.tools.anomaly_detection import detect_anomalies, get_anomaly_summary
            result = detect_anomalies(st.session_state.data, date_col='date', value_col='revenue', method='zscore')
            st.session_state.analysis_results["anomaly"] = result
            
            count = result.get('count', 0)
            method = result.get('method', 'unknown')
            rate = result.get('anomaly_rate', 0) * 100
            
            reply = (
                "Anomaly detection summary:\n"
                f"- Total anomalies: {count}\n"
                f"- Anomaly rate: {rate:.2f}%\n"
                f"- Method: {method.upper()}\n"
            )
            st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            st.session_state.chat_history.append({'role': 'assistant', 'content': f"Error detecting anomalies: {e}"})
    
    def qa_simulate_promo():
        """Fast promotion simulation without LLM."""
        try:
            from src.tools.scenario_simulation import simulate_promotion
            result = simulate_promotion(st.session_state.data, discount_pct=0.15, duration_days=7)
            st.session_state.analysis_results["scenario"] = result
            
            baseline = result.get('baseline', {})
            projected = result.get('projected', {})
            impact = result.get('impact', {})
            
            revenue_lift = impact.get('revenue_lift_pct', 0)
            net_revenue = impact.get('net_incremental_revenue', 0)
            
            reply = (
                "Promotion simulation (15% discount for 7 days):\n"
                f"- Baseline revenue: ${baseline.get('revenue', 0):,.0f}\n"
                f"- Projected revenue: ${projected.get('revenue', 0):,.0f}\n"
                f"- Revenue lift: {revenue_lift:+.1f}%\n"
                f"- Net incremental revenue: ${net_revenue:,.0f}\n"
                f"- Recommendation: {result.get('recommendation', 'N/A')}"
            )
            st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            st.session_state.chat_history.append({'role': 'assistant', 'content': f"Error simulating promotion: {e}"})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Show Trends", use_container_width=True):
            qa_show_trends()
            st.rerun()
    
    with col2:
        if st.button("🔍 Find Anomalies", use_container_width=True):
            qa_find_anomalies()
            st.rerun()
    
    with col3:
        if st.button("🎯 Simulate Promo", use_container_width=True):
            qa_simulate_promo()
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.memory.clear()
            st.rerun()

# -----------------------
# Analytics Page
# -----------------------

def render_analytics():
    """Render interactive analytics page with modern visualizations and real-time updates."""
    st.markdown("## 📈 Interactive Analytics Dashboard")
    st.caption("Advanced analytics with real-time filtering and interactive visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    # Global Filters Section (applies to all tabs)
    st.markdown("### 🎛️ Global Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        if 'date' in data.columns:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            
            # Calculate default range - always use this as the default value
            # The widget will manage its own session state via the key
            # Don't read from or write to session state before the widget
            default_range = (min_date, max_date)
            
            try:
                # Create widget with default value
                # The widget manages its own session state via the key parameter
                # This avoids the warning about setting session state before widget creation
                date_range = st.date_input(
                    "📅 Date Range",
                    value=default_range,
                    min_value=min_date,
                    max_value=max_date,
                    key="analytics_date_range"
                )
                
                # Use the widget's returned value (widget manages its own session state)
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    # Clamp to valid range
                    if start_date < min_date:
                        start_date = min_date
                    if end_date > max_date:
                        end_date = max_date
                    if start_date > end_date:
                        start_date = min_date
                        end_date = max_date
                    
                    # Filter data
                    data = data[(data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)]
                    
                    # Show info if dates were clamped
                    if start_date != date_range[0] or end_date != date_range[1]:
                        st.caption(f"ℹ️ Date range adjusted to available data: {start_date} to {end_date}")
                elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
                    # Single date selected - use as both start and end
                    single_date = date_range[0] if isinstance(date_range, tuple) else date_range[0]
                    if single_date < min_date:
                        single_date = min_date
                    elif single_date > max_date:
                        single_date = max_date
                    data = data[data['date'].dt.date == single_date]
            except Exception as e:
                st.caption(f"⚠️ Date range error: {str(e)}. Using available data: {min_date} to {max_date}")
                data = data[(data['date'].dt.date >= min_date) & (data['date'].dt.date <= max_date)]
    
    with filter_col2:
        if 'category' in data.columns:
            categories = ['All'] + sorted(data['category'].unique().tolist())
            selected_category = st.selectbox("📦 Category", categories, key="analytics_category")
            if selected_category != 'All':
                data = data[data['category'] == selected_category]
    
    with filter_col3:
        if 'store_region' in data.columns:
            regions = ['All'] + sorted(data['store_region'].unique().tolist())
            selected_region = st.selectbox("🗺️ Region", regions, key="analytics_region")
            if selected_region != 'All':
                data = data[data['store_region'] == selected_region]
    
    st.markdown("---")
    
    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🔍 Anomaly Detection", "🎯 Scenario Simulation"])
    
    # Trend Analysis Tab
    with tab1:
        st.markdown("### 📈 Advanced Trend Analysis")
        
        # Configuration
        config_col1, config_col2, config_col3, config_col4 = st.columns(4)
        with config_col1:
            metric = st.selectbox("📊 Metric", ["revenue", "units_sold", "price"], key="trend_metric")
        with config_col2:
            period = st.selectbox("📅 Period", ["daily", "weekly", "monthly"], key="trend_period", index=2)
        with config_col3:
            show_seasonality = st.checkbox("🌊 Show Seasonality", value=False, key="trend_seasonality")
        with config_col4:
            show_forecast = st.checkbox("🔮 Show Forecast", value=False, key="trend_forecast")
        
        # Auto-analyze when filters change
        try:
            with st.spinner("Analyzing trends..."):
                result = extract_trends(data, date_col='date', value_col=metric, period='daily')  # Use daily for trend
                # Also calculate growth rate (matching reference - uses monthly)
                growth_result = calculate_growth_rate(data, date_col='date', value_col=metric, period='monthly', method='compound')
                result['growth_rate'] = growth_result.get('growth_rate', 0)
                st.session_state.analysis_results['trend'] = result
        except Exception as e:
            st.error(f"Error: {e}")
            result = None
        
        if result and 'trend' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['trend']
            
            # Enhanced Metrics Display (matching reference format)
            st.markdown("#### 📊 Trend Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trend = result.get('trend_direction', result.get('trend', 'unknown'))
                icon = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                st.markdown(f"**Trend Direction**\n\n{icon} {trend.title()}")
            
            with col2:
                growth_rate = result.get('growth_rate', 0)
                st.markdown(f"**Growth Rate**\n\n{growth_rate:.1f}%")
            
            with col3:
                r2 = result.get('r_squared', 0)
                if r2 is None or pd.isna(r2):
                    r2 = 0
                st.markdown(f"**R² Score**\n\n{r2:.3f}")
            
            with col4:
                is_sig = result.get('is_significant', False)
                sig_text = "Yes ✓" if is_sig else "No ✗"
                st.markdown(f"**Significant**\n\n{sig_text}")
            
            st.markdown("---")
            
            # Advanced Interactive Visualization
            if 'date' in data.columns:
                data_copy = data.copy()
                data_copy['date'] = pd.to_datetime(data_copy['date'])
                
                # Aggregate by period
                if period == 'daily':
                    agg = data_copy.groupby('date')[metric].sum().reset_index()
                elif period == 'weekly':
                    data_copy['period'] = data_copy['date'] - pd.to_timedelta(data_copy['date'].dt.dayofweek, unit='d')
                    agg = data_copy.groupby('period')[metric].sum().reset_index().rename(columns={'period': 'date'})
                else:
                    data_copy['period'] = data_copy['date'].dt.to_period('M').dt.to_timestamp()
                    agg = data_copy.groupby('period')[metric].sum().reset_index().rename(columns={'period': 'date'})
                
                # Create interactive figure
                fig = go.Figure()
                
                # Actual data with enhanced styling
                fig.add_trace(go.Scatter(
                    x=agg['date'],
                    y=agg[metric],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=6, color='#667eea'),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: %{y:,.0f}<extra></extra>'
                ))
                
                # Trend line
                slope_val = result.get('slope', 0)
                intercept_val = result.get('intercept', 0)
                xnum = np.arange(len(agg))
                trend_line = intercept_val + slope_val * xnum
                
                fig.add_trace(go.Scatter(
                    x=agg['date'],
                    y=trend_line,
                    mode='lines',
                    name='Linear Trend',
                    line=dict(color='#f5576c', width=3, dash='dash'),
                    hovertemplate='<b>Trend</b><br>%{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>'
                ))
                
                # Moving average
                if len(agg) > 7:
                    window = min(7, len(agg) // 4)
                    agg['ma'] = agg[metric].rolling(window=window, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=agg['date'],
                        y=agg['ma'],
                        mode='lines',
                        name=f'{window}-Period MA',
                        line=dict(color='#48bb78', width=2, dash='dot'),
                        hovertemplate='<b>Moving Avg</b><br>%{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>'
                    ))
                
                # Forecast (if enabled)
                if show_forecast and len(agg) > 5:
                    future_periods = 3
                    last_date = agg['date'].iloc[-1]
                    if period == 'daily':
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods, freq='D')
                    elif period == 'weekly':
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_periods, freq='W')
                    else:
                        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
                    
                    future_xnum = np.arange(len(agg), len(agg) + future_periods)
                    future_trend = intercept_val + slope_val * future_xnum
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_trend,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#fbbf24', width=2, dash='dot'),
                        marker=dict(size=8, symbol='diamond'),
                        hovertemplate='<b>Forecast</b><br>%{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>'
                    ))
                
                # Confidence interval
                if len(agg) > 10:
                    std_dev = agg[metric].std()
                    upper_bound = trend_line + 1.96 * std_dev
                    lower_bound = trend_line - 1.96 * std_dev
                    
                    fig.add_trace(go.Scatter(
                        x=agg['date'],
                        y=upper_bound,
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=agg['date'],
                        y=lower_bound,
                        mode='lines',
                        name='95% CI',
                        fill='tonexty',
                        fillcolor='rgba(245, 87, 108, 0.1)',
                        line=dict(width=0),
                        hovertemplate='<b>95% CI</b><extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"{metric.replace('_',' ').title()} Trend Analysis ({period.title()})",
                    xaxis_title="Date",
                    yaxis_title=metric.replace('_',' ').title(),
                    height=st.session_state.chart_height,
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                config = {'displayModeBar': True, 'displaylogo': False}
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Seasonality visualization
                if show_seasonality and len(agg) > 30:
                    st.markdown("---")
                    st.markdown("#### 🌊 Seasonal Pattern Analysis")
                    agg['month'] = pd.to_datetime(agg['date']).dt.month
                    seasonal_pattern = agg.groupby('month')[metric].mean()
                    
                    seasonal_fig = go.Figure()
                    seasonal_fig.add_trace(go.Bar(
                        x=seasonal_pattern.index,
                        y=seasonal_pattern.values,
                        marker=dict(
                            color=seasonal_pattern.values,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Avg by Month',
                        hovertemplate='<b>Month %{x}</b><br>Avg: %{y:,.0f}<extra></extra>'
                    ))
                    seasonal_fig.update_layout(
                        title='Seasonal Pattern (Average by Month)',
                        xaxis_title='Month',
                        yaxis_title=f'Average {metric.replace("_", " ").title()}',
                        height=350,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(seasonal_fig, use_container_width=True)
                
                # Insights section
                st.markdown("---")
                st.markdown("#### 💡 Insights & Recommendations")
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown("**Trend Analysis:**")
                    # Extract variables from result
                    trend = result.get('trend_direction', result.get('trend', 'unknown'))
                    slope_val = result.get('slope', 0)
                    strength = result.get('trend_strength', 'unknown')
                    
                    if trend == 'increasing':
                        st.success(f"✅ **{metric.replace('_', ' ').title()}** is showing an **increasing trend** with **{strength}** strength.")
                        if slope_val > 0:
                            st.info(f"📊 The trend is growing at a rate of **{slope_val:.4f}** per period.")
                    elif trend == 'decreasing':
                        st.warning(f"⚠️ **{metric.replace('_', ' ').title()}** is showing a **decreasing trend** with **{strength}** strength.")
                        st.info(f"📊 The trend is declining at a rate of **{abs(slope_val):.4f}** per period.")
                    else:
                        st.info(f"ℹ️ **{metric.replace('_', ' ').title()}** is showing a **stable trend**.")
                
                with insight_col2:
                    st.markdown("**Model Quality:**")
                    r2_value = result.get('r_squared', 0)
                    # Handle None, NaN, or invalid values
                    try:
                        if r2_value is None or (isinstance(r2_value, float) and pd.isna(r2_value)):
                            r2_value = 0.0
                        r2_value = float(r2_value)
                        r2_str = f"{r2_value:.3f}"
                    except (ValueError, TypeError):
                        r2_value = 0.0
                        r2_str = "0.000"
                    
                    if r2_value > 0.7:
                        st.success(f"✅ **Strong correlation** (R² = {r2_str}) - Trend model is highly reliable.")
                    elif r2_value > 0.3:
                        st.info(f"ℹ️ **Moderate correlation** (R² = {r2_str}) - Trend model has moderate reliability.")
                    else:
                        st.warning(f"⚠️ **Weak correlation** (R² = {r2_str}) - Trend may not be statistically significant.")
    
    # Anomaly Detection Tab
    with tab2:
        st.markdown("### 🔍 Advanced Anomaly Detection")
        
        # Configuration
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            metric = st.selectbox("📊 Metric", ["revenue", "units_sold", "price"], key="anomaly_metric")
        with config_col2:
            include_multivariate = st.checkbox("🔬 Include Multivariate Detection", value=True, key="anomaly_multivariate")
        
        # Auto-detect when filters change (using get_anomaly_summary matching reference)
        try:
            with st.spinner("Detecting anomalies..."):
                result = get_anomaly_summary(data, metric=metric, include_multivariate=include_multivariate)
                st.session_state.analysis_results['anomaly'] = result
        except Exception as e:
            st.error(f"Error: {e}")
            result = None
        
        if result and 'anomaly' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['anomaly']
            
            # Check if it's the new format (from get_anomaly_summary)
            if 'overall_assessment' in result:
                # New comprehensive format matching reference
                assess = result['overall_assessment']
                
                # Data quality indicator
                if assess['data_quality'] == 'good':
                    st.success(f"✓ Data Quality: {assess['data_quality'].upper()} | Anomaly Rate: {assess['anomaly_rate']:.2f}%")
                else:
                    st.warning(f"⚠️ Data Quality: {assess['data_quality'].upper()} | Anomaly Rate: {assess['anomaly_rate']:.2f}%")
                
                # Total Anomalies Metrics (matching reference layout)
                st.markdown("#### 📊 Anomaly Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Anomalies", assess['total_anomalies'])
                with col2:
                    st.metric("Statistical Outliers", result.get('statistical_outliers', {}).get('count', 0))
                with col3:
                    st.metric("Time Series Anomalies", result.get('time_series_anomalies', {}).get('count', 0))
                with col4:
                    st.metric("Multivariate Anomalies", result.get('multivariate_anomalies', {}).get('count', 0))
                
                # Anomaly Breakdown (matching reference)
                st.markdown("### 📊 Anomaly Breakdown")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Statistical Outliers:**")
                    stat_outliers = result.get('statistical_outliers', {})
                    st.write(f"- High outliers: {stat_outliers.get('high_outliers', 0)}")
                    st.write(f"- Low outliers: {stat_outliers.get('low_outliers', 0)}")
                with c2:
                    st.markdown("**Time Series Anomalies:**")
                    ts_anomalies = result.get('time_series_anomalies', {})
                    st.write(f"- Spikes: {ts_anomalies.get('spikes', 0)}")
                    st.write(f"- Drops: {ts_anomalies.get('drops', 0)}")
                
                # Use statistical outliers for visualization (most relevant)
                anomalies_list = result['statistical_outliers'].get('examples', [])
            else:
                # Old format (backward compatibility)
                anomalies_list = result.get('anomalies', [])
                if not isinstance(anomalies_list, list):
                    anomalies_list = []
                
                # Enhanced Metrics
                st.markdown("#### 📊 Anomaly Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    anomaly_count = result.get('count', len(anomalies_list))
                    total_points = len(data)
                    anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
                    st.metric("Total Anomalies", f"{anomaly_count:,}", f"{anomaly_rate:.1f}%")
                
                with col2:
                    if anomalies_list and 'date' in data.columns:
                        mean_val = data[metric].mean()
                        high_anomalies = sum(1 for a in anomalies_list if a.get('value', 0) > mean_val)
                        st.metric("High Anomalies", high_anomalies, "Above mean")
                    else:
                        st.metric("High Anomalies", 0)
                
                with col3:
                    if anomalies_list and 'date' in data.columns:
                        mean_val = data[metric].mean()
                        low_anomalies = sum(1 for a in anomalies_list if a.get('value', 0) < mean_val)
                        st.metric("Low Anomalies", low_anomalies, "Below mean")
                    else:
                        st.metric("Low Anomalies", 0)
                
                with col4:
                    method_used = result.get('method', 'unknown')
                    st.metric("Method", method_used.upper())
                
                with col5:
                    if anomalies_list:
                        avg_anomaly_value = np.mean([a.get('value', 0) for a in anomalies_list])
                        st.metric("Avg Anomaly", f"{avg_anomaly_value:,.0f}")
                    else:
                        st.metric("Avg Anomaly", "N/A")
            
            st.markdown("---")
            
            # Visualization
            if 'date' in data.columns and 'anomalies' in result:
                anomalies_list = result.get('anomalies', [])
                # Ensure anomalies_list is actually a list
                if not isinstance(anomalies_list, list):
                    anomalies_list = []
                
                if anomalies_list and len(anomalies_list) > 0:
                    try:
                        # Convert anomalies list to DataFrame for easier handling
                        anomalies_df = pd.DataFrame(anomalies_list)
                        
                        # Verify DataFrame was created successfully
                        if isinstance(anomalies_df, pd.DataFrame) and len(anomalies_df) > 0:
                            fig = go.Figure()
                            # Plot all data points
                            fig.add_trace(go.Scatter(
                                x=data['date'],
                                y=data[metric],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='steelblue', size=4, opacity=0.6)
                            ))
                            # Plot anomaly points
                            if 'date' in anomalies_df.columns and 'value' in anomalies_df.columns:
                                fig.add_trace(go.Scatter(
                                    x=anomalies_df['date'],
                                    y=anomalies_df['value'],
                                    mode='markers',
                                    name='Anomaly',
                                    marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
                                ))
                            fig.update_layout(
                                title=f"Anomaly Detection: {metric.replace('_',' ').title()}",
                                xaxis_title="Date",
                                yaxis_title=metric.replace('_',' ').title(),
                                height=st.session_state.chart_height,
                                hovermode='closest'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not create visualization from anomaly data.")
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
                        st.json(anomalies_list[:5])  # Show first 5 anomalies for debugging
                else:
                    st.info("No anomalies detected with the current settings.")
    
    # Scenario Simulation Tab
    with tab3:
        st.markdown("### 🎯 Scenario Simulation")
        
        scenario_type = st.radio(
            "Select Scenario Type",
            ["Promotion", "Price Change"],
            horizontal=True,
            key="scenario_type"
        )
        
        st.markdown("---")
        
        if scenario_type == "Promotion":
            st.markdown("#### 🎁 Promotion Simulation")
            
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                discount = st.slider("💸 Discount (%)", 5, 50, 20, 1, key="promo_discount")
            with sim_col2:
                duration = st.slider("⏱️ Duration (days)", 1, 30, 7, 1, key="promo_duration")
            with sim_col3:
                expected_lift = st.slider("📈 Expected Lift (%)", 10, 200, 50, 5, key="promo_lift")
            
            # Auto-simulate
            try:
                with st.spinner("Running simulation..."):
                    result = simulate_promotion(data, discount_pct=discount / 100.0, duration_days=duration)
                    st.session_state.analysis_results['scenario'] = result
            except Exception as e:
                st.error(f"Error: {e}")
                result = None
        
        else:  # Price Change
            st.markdown("#### 💰 Price Change Simulation")
            
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                price_change = st.slider("💵 Price Change (%)", -30, 30, 10, 1, key="price_change")
            with sim_col2:
                use_custom_elasticity = st.checkbox("Use custom elasticity", value=False, key="use_custom_elasticity")
                if use_custom_elasticity:
                    elasticity = st.slider("📊 Price Elasticity", -3.0, 3.0, -1.5, 0.1, key="price_elasticity")
                else:
                    elasticity = None  # Use default -1.5 matching reference
            
            # Auto-simulate
            try:
                with st.spinner("Running simulation..."):
                    # Pass price_change as percentage (matching reference implementation)
                    # If elasticity is None, function will use default -1.5
                    result = simulate_price_change(
                        data, 
                        price_change_pct=price_change,  # Pass as percentage (10.0 for 10%)
                        price_elasticity=elasticity if use_custom_elasticity else None
                    )
                    st.session_state.analysis_results['scenario'] = result
            except Exception as e:
                st.error(f"Error: {e}")
                result = None
        
        if result and 'scenario' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['scenario']
            
            st.markdown("---")
            st.markdown("#### 📊 Simulation Results")
            
            # Enhanced Metrics Display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
                    <strong style="color: #667eea; font-size: 1.1rem;">📈 Baseline</strong>
                </div>
                """, unsafe_allow_html=True)
                baseline = result.get('baseline', {})
                baseline_rev = baseline.get('revenue', 0)
                baseline_units = baseline.get('units_sold', baseline.get('units', 0))
                st.metric("Revenue", f"${baseline_rev:,.0f}")
                st.metric("Units", f"{baseline_units:,.0f}")
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #48bb78;">
                    <strong style="color: #48bb78; font-size: 1.1rem;">🚀 Projected</strong>
                </div>
                """, unsafe_allow_html=True)
                simulated = result.get('simulated', result.get('projected', {}))
                projected_rev = simulated.get('revenue', 0)
                projected_units = simulated.get('units_sold', simulated.get('units', 0))
                st.metric("Revenue", f"${projected_rev:,.0f}")
                st.metric("Units", f"{projected_units:,.0f}")
            
            with col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #fbbf24;">
                    <strong style="color: #fbbf24; font-size: 1.1rem;">📊 Impact</strong>
                </div>
                """, unsafe_allow_html=True)
                impact = result.get('impact', {})
                revenue_change_pct = impact.get('revenue_lift_pct', 0) if 'revenue_lift_pct' in impact else impact.get('revenue_change_pct', 0)
                units_change_pct = impact.get('units_change_pct', 0) if 'units_change_pct' in impact else 0
                revenue_delta = f"{revenue_change_pct:+.1f}%"
                units_delta = f"{units_change_pct:+.1f}%"
                st.metric("Revenue Change", revenue_delta, delta=revenue_delta)
                st.metric("Units Change", units_delta, delta=units_delta)
            
            with col4:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #f5576c;">
                    <strong style="color: #f5576c; font-size: 1.1rem;">💵 Difference</strong>
                </div>
                """, unsafe_allow_html=True)
                rev_diff = projected_rev - baseline_rev
                units_diff = projected_units - baseline_units
                st.metric("Revenue Δ", f"${rev_diff:+,.0f}")
                st.metric("Units Δ", f"{units_diff:+,.0f}")
            
            # Visualization
            st.markdown("---")
            st.markdown("#### 📈 Comparison Visualization")
            
            comparison_fig = go.Figure()
            
            # Baseline bar
            comparison_fig.add_trace(go.Bar(
                name='Baseline',
                x=['Revenue', 'Units'],
                y=[baseline_rev / 1000, baseline_units / 1000],  # Scale for visibility
                marker_color='#667eea',
                text=[f"${baseline_rev:,.0f}", f"{baseline_units:,.0f}"],
                textposition='outside',
                hovertemplate='<b>Baseline</b><br>%{x}: %{text}<extra></extra>'
            ))
            
            # Projected bar
            comparison_fig.add_trace(go.Bar(
                name='Projected',
                x=['Revenue', 'Units'],
                y=[projected_rev / 1000, projected_units / 1000],
                marker_color='#48bb78',
                text=[f"${projected_rev:,.0f}", f"{projected_units:,.0f}"],
                textposition='outside',
                hovertemplate='<b>Projected</b><br>%{x}: %{text}<extra></extra>'
            ))
            
            comparison_fig.update_layout(
                title='Baseline vs Projected Comparison',
                yaxis_title='Value (scaled)',
                height=400,
                template='plotly_white',
                barmode='group',
                showlegend=True
            )
            
            config = {'displayModeBar': True, 'displaylogo': False}
            st.plotly_chart(comparison_fig, use_container_width=True, config=config)
            
            # Recommendation
            st.markdown("---")
            if 'recommendation' in result:
                st.markdown("#### 💡 Recommendation")
                st.info(f"**{result['recommendation']}**")
            
            # Additional insights
            if revenue_change_pct > 0:
                st.success(f"✅ This scenario shows a **positive impact** with {revenue_change_pct:+.1f}% revenue change.")
            elif revenue_change_pct < 0:
                st.warning(f"⚠️ This scenario shows a **negative impact** with {revenue_change_pct:+.1f}% revenue change.")
            else:
                st.info(f"ℹ️ This scenario shows **neutral impact** on revenue.")

# -----------------------
# Dashboard Page
# -----------------------

def render_dashboard():
    """Render comprehensive interactive dashboard with modern visualizations."""
    st.markdown("## 📊 Interactive Executive Dashboard")
    st.caption("Explore your sales data with interactive filters and advanced visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    # Interactive Filters Section
    st.markdown("### 🎛️ Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        # Date range filter
        if 'date' in data.columns:
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            date_range = st.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="dashboard_date_range"
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                data = data[(data['date'].dt.date >= date_range[0]) & (data['date'].dt.date <= date_range[1])]
    
    with filter_col2:
        # Category filter
        if 'category' in data.columns:
            categories = ['All'] + sorted(data['category'].unique().tolist())
            selected_category = st.selectbox("📦 Category", categories, key="dashboard_category")
            if selected_category != 'All':
                data = data[data['category'] == selected_category]
    
    with filter_col3:
        # Region filter
        if 'store_region' in data.columns:
            regions = ['All'] + sorted(data['store_region'].unique().tolist())
            selected_region = st.selectbox("🗺️ Region", regions, key="dashboard_region")
            if selected_region != 'All':
                data = data[data['store_region'] == selected_region]
    
    with filter_col4:
        # Aggregation period
        agg_period = st.selectbox("📊 Period", ["Daily", "Weekly", "Monthly"], key="dashboard_period", index=2)
    
    st.markdown("---")
    
    # Key Performance Indicators (KPIs) with trends
    st.markdown("### 📈 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        if 'revenue' in data.columns:
            total_revenue = data['revenue'].sum()
            # Calculate trend (compare first half vs second half)
            if len(data) > 1 and 'date' in data.columns:
                mid_point = len(data) // 2
                first_half = data.iloc[:mid_point]['revenue'].sum()
                second_half = data.iloc[mid_point:]['revenue'].sum()
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                delta = f"{trend:+.1f}%"
            else:
                delta = None
            st.metric("Total Revenue", f"${total_revenue:,.0f}", delta=delta)
    
    with kpi_col2:
        if 'units_sold' in data.columns:
            total_units = data['units_sold'].sum()
            if len(data) > 1 and 'date' in data.columns:
                mid_point = len(data) // 2
                first_half = data.iloc[:mid_point]['units_sold'].sum()
                second_half = data.iloc[mid_point:]['units_sold'].sum()
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                delta = f"{trend:+.1f}%"
            else:
                delta = None
            st.metric("Total Units", f"{total_units:,.0f}", delta=delta)
    
    with kpi_col3:
        if 'price' in data.columns and 'revenue' in data.columns and 'units_sold' in data.columns:
            # Calculate average price per unit correctly: total revenue / total units
            # This matches the scenario simulation calculation for consistency
            total_revenue = data['revenue'].sum()
            total_units = data['units_sold'].sum()
            avg_price = total_revenue / total_units if total_units > 0 else data['price'].mean()
            
            if len(data) > 1 and 'date' in data.columns:
                mid_point = len(data) // 2
                first_half_rev = data.iloc[:mid_point]['revenue'].sum()
                first_half_units = data.iloc[:mid_point]['units_sold'].sum()
                second_half_rev = data.iloc[mid_point:]['revenue'].sum()
                second_half_units = data.iloc[mid_point:]['units_sold'].sum()
                first_half_price = first_half_rev / first_half_units if first_half_units > 0 else 0
                second_half_price = second_half_rev / second_half_units if second_half_units > 0 else 0
                trend = ((second_half_price - first_half_price) / first_half_price * 100) if first_half_price > 0 else 0
                delta = f"{trend:+.1f}%"
            else:
                delta = None
            st.metric("Avg Price", f"${avg_price:.2f}", delta=delta)
        elif 'price' in data.columns:
            # Fallback if units_sold not available
            avg_price = data['price'].mean()
            if len(data) > 1 and 'date' in data.columns:
                mid_point = len(data) // 2
                first_half = data.iloc[:mid_point]['price'].mean()
                second_half = data.iloc[mid_point:]['price'].mean()
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                delta = f"{trend:+.1f}%"
            else:
                delta = None
            st.metric("Avg Price", f"${avg_price:.2f}", delta=delta)
    
    with kpi_col4:
        if 'promo_flag' in data.columns:
            promo_rate = data['promo_flag'].mean() * 100
            promo_count = data['promo_flag'].sum()
            st.metric("Promo Rate", f"{promo_rate:.1f}%", f"{promo_count:,} transactions")
    
    with kpi_col5:
        if 'date' in data.columns:
            unique_days = data['date'].nunique()
            total_transactions = len(data)
            st.metric("Transactions", f"{total_transactions:,}", f"{unique_days} days")
    
    st.markdown("---")
    
    # Main Visualizations
    # Revenue over time with multiple metrics
    st.markdown("### 📊 Revenue & Sales Trends")
    
    if 'date' in data.columns and 'revenue' in data.columns:
        # Aggregate by selected period
        data['date'] = pd.to_datetime(data['date'])
        if agg_period == "Daily":
            time_agg = data.groupby('date').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
        elif agg_period == "Weekly":
            data['week'] = data['date'] - pd.to_timedelta(data['date'].dt.dayofweek, unit='d')
            time_agg = data.groupby('week').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
            time_agg.rename(columns={'week': 'date'}, inplace=True)
        else:  # Monthly
            data['month'] = data['date'].dt.to_period('M').dt.to_timestamp()
            time_agg = data.groupby('month').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
            time_agg.rename(columns={'month': 'date'}, inplace=True)
        
        # Create dual-axis chart
        fig = go.Figure()
        
        # Revenue line
        fig.add_trace(go.Scatter(
            x=time_agg['date'],
            y=time_agg['revenue'],
            name='Revenue',
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6),
            yaxis='y',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        # Units sold (secondary axis)
        if 'units_sold' in time_agg.columns:
            fig.add_trace(go.Bar(
                x=time_agg['date'],
                y=time_agg['units_sold'],
                name='Units Sold',
                marker_color='rgba(118, 75, 162, 0.6)',
                yaxis='y2',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Units: %{y:,.0f}<extra></extra>'
            ))
        
        # Add moving average
        if len(time_agg) > 7:
            window = min(7, len(time_agg) // 4)
            time_agg['revenue_ma'] = time_agg['revenue'].rolling(window=window, center=True).mean()
            fig.add_trace(go.Scatter(
                x=time_agg['date'],
                y=time_agg['revenue_ma'],
                name=f'{window}-Period Moving Avg',
                mode='lines',
                line=dict(color='#f5576c', width=2, dash='dash'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MA: $%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Revenue & Sales Trend ({agg_period})',
            xaxis_title='Date',
            yaxis=dict(title='Revenue ($)', side='left'),
            yaxis2=dict(title='Units Sold', side='right', overlaying='y'),
            height=st.session_state.chart_height,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Use config parameter for Plotly configuration
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
        st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Category and Region Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Category Performance")
        if 'category' in data.columns and 'revenue' in data.columns:
            cat_analysis = data.groupby('category').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
            # Calculate average price per unit correctly: revenue / units
            if 'units_sold' in cat_analysis.columns and cat_analysis['units_sold'].sum() > 0:
                cat_analysis['price'] = cat_analysis['revenue'] / cat_analysis['units_sold']
            elif 'price' in data.columns:
                # Fallback to mean if units not available
                cat_analysis['price'] = data.groupby('category')['price'].mean().values
            else:
                cat_analysis['price'] = 0
            cat_analysis = cat_analysis.sort_values('revenue', ascending=False).head(10)
            
            # Interactive bar chart with hover info
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_analysis['revenue'],
                y=cat_analysis['category'],
                orientation='h',
                marker=dict(
                    color=cat_analysis['revenue'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Revenue")
                ),
                text=[f"${x:,.0f}" for x in cat_analysis['revenue']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Units: %{customdata:,.0f}<extra></extra>',
                customdata=cat_analysis['units_sold'] if 'units_sold' in cat_analysis.columns else [0]*len(cat_analysis)
            ))
            
            fig.update_layout(
                title='Top 10 Categories by Revenue',
                xaxis_title='Revenue ($)',
                yaxis_title='Category',
                height=st.session_state.chart_height,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Regional Analysis")
        if 'store_region' in data.columns and 'revenue' in data.columns:
            region_analysis = data.groupby('store_region').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
            
            # Interactive pie chart with donut style
            fig = go.Figure(data=[go.Pie(
                labels=region_analysis['store_region'],
                values=region_analysis['revenue'],
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(colors=px.colors.qualitative.Set3),
                hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Revenue Distribution by Region',
                height=st.session_state.chart_height,
                template='plotly_white',
                annotations=[dict(text='Revenue', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Visualizations
    st.markdown("---")
    st.markdown("### 🔍 Advanced Analytics")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.markdown("#### 📊 Revenue Heatmap (by Category & Region)")
        if 'category' in data.columns and 'store_region' in data.columns and 'revenue' in data.columns:
            heatmap_data = data.pivot_table(
                values='revenue',
                index='category',
                columns='store_region',
                aggfunc='sum',
                fill_value=0
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='YlOrRd',
                text=[[f"${val:,.0f}" for val in row] for row in heatmap_data.values],
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='<b>%{y} - %{x}</b><br>Revenue: $%{z:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Revenue Heatmap: Category vs Region',
                height=400,
                template='plotly_white',
                xaxis_title='Region',
                yaxis_title='Category'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with adv_col2:
        st.markdown("#### 💰 Price vs Revenue Scatter")
        if 'price' in data.columns and 'revenue' in data.columns:
            # Aggregate by category for cleaner visualization
            if 'category' in data.columns:
                scatter_data = data.groupby('category').agg({
                    'revenue': 'sum',
                    'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
                }).reset_index()
                # Calculate average price per unit correctly: revenue / units
                if 'units_sold' in scatter_data.columns and scatter_data['units_sold'].sum() > 0:
                    scatter_data['price'] = scatter_data['revenue'] / scatter_data['units_sold']
                elif 'price' in data.columns:
                    # Fallback to mean if units not available
                    scatter_data['price'] = data.groupby('category')['price'].mean().values
                else:
                    scatter_data['price'] = 0
                
                fig = px.scatter(
                    scatter_data,
                    x='price',
                    y='revenue',
                    size='units_sold' if 'units_sold' in scatter_data.columns else None,
                    color='category',
                    hover_name='category',
                    hover_data={'price': ':.2f', 'revenue': ':,.0f'},
                    title='Price vs Revenue by Category',
                    labels={'price': 'Average Price ($)', 'revenue': 'Total Revenue ($)'}
                )
                
                fig.update_layout(
                    height=400,
                    template='plotly_white',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Performance Comparison
    st.markdown("---")
    st.markdown("### 📈 Performance Comparison")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("#### 🏪 Store Performance")
        if 'store_id' in data.columns and 'revenue' in data.columns:
            store_perf = data.groupby('store_id').agg({
                'revenue': 'sum',
                'units_sold': 'sum' if 'units_sold' in data.columns else 'count'
            }).reset_index()
            store_perf = store_perf.sort_values('revenue', ascending=False).head(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=store_perf['store_id'].astype(str),
                y=store_perf['revenue'],
                name='Revenue',
                marker_color='#667eea',
                hovertemplate='<b>Store %{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top 15 Stores by Revenue',
                xaxis_title='Store ID',
                yaxis_title='Revenue ($)',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with comp_col2:
        st.markdown("#### 📅 Day of Week Analysis")
        if 'date' in data.columns and 'revenue' in data.columns:
            data['day_of_week'] = data['date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_perf = data.groupby('day_of_week')['revenue'].sum().reindex(day_order, fill_value=0).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=day_perf['day_of_week'],
                y=day_perf['revenue'],
                marker=dict(
                    color=day_perf['revenue'],
                    colorscale='Blues',
                    showscale=True
                ),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Revenue by Day of Week',
                xaxis_title='Day',
                yaxis_title='Revenue ($)',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics Table
    st.markdown("---")
    st.markdown("### 📋 Summary Statistics")
    
    if len(data) > 0:
        summary_stats = []
        
        numeric_cols = ['revenue', 'units_sold', 'price']
        for col in numeric_cols:
            if col in data.columns:
                # For price, calculate average price per unit (revenue/units) for consistency
                # This matches the scenario simulation and KPI calculations
                if col == 'price' and 'revenue' in data.columns and 'units_sold' in data.columns:
                    total_rev = data['revenue'].sum()
                    total_units = data['units_sold'].sum()
                    avg_price_per_unit = total_rev / total_units if total_units > 0 else data[col].mean()
                    mean_val = avg_price_per_unit
                    # For median, std, min, max, use transaction price (for distribution info)
                    median_val = data[col].median()
                    std_val = data[col].std()
                    min_val = data[col].min()
                    max_val = data[col].max()
                else:
                    mean_val = data[col].mean()
                    median_val = data[col].median()
                    std_val = data[col].std()
                    min_val = data[col].min()
                    max_val = data[col].max()
                
                summary_stats.append({
                    'Metric': col.replace('_', ' ').title(),
                    'Total': f"${data[col].sum():,.2f}" if col == 'revenue' else f"{data[col].sum():,.0f}",
                    'Mean': f"${mean_val:,.2f}" if col in ['revenue', 'price'] else f"{mean_val:,.2f}",
                    'Median': f"${median_val:,.2f}" if col in ['revenue', 'price'] else f"{median_val:,.2f}",
                    'Std Dev': f"${std_val:,.2f}" if col in ['revenue', 'price'] else f"{std_val:,.2f}",
                    'Min': f"${min_val:,.2f}" if col in ['revenue', 'price'] else f"{min_val:,.2f}",
                    'Max': f"${max_val:,.2f}" if col in ['revenue', 'price'] else f"{max_val:,.2f}"
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            # Wrap in styled container for better visibility
            st.markdown("""
            <div class="summary-card">
                <h4 style="color: #667eea; margin-bottom: 1rem;">Statistical Overview</h4>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# -----------------------
# Chat History Page
# -----------------------

def render_chat_history():
    """Render chat history browser with advanced features."""
    st.markdown("## 🕘 Chat History Browser")
    st.caption("View, search, export, and manage your conversation history")
    
    # Get all messages
    ui_messages = st.session_state.get('chat_history', [])
    agent_messages = []
    
    if st.session_state.agent and hasattr(st.session_state.agent, 'memory'):
        agent_messages = st.session_state.agent.memory.conversation_history
    
    # Combine messages
    all_messages = []
    for msg in ui_messages:
        all_messages.append({
            'role': msg.get('role', 'unknown'),
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', datetime.now().isoformat()),
            'source': 'UI Chat'
        })
    
    for msg in agent_messages:
        all_messages.append({
            'role': msg.get('role', 'unknown'),
            'content': str(msg.get('content', '')),
            'timestamp': msg.get('timestamp', datetime.now().isoformat()),
            'source': 'Agent Memory'
        })
    
    all_messages.sort(key=lambda x: x.get('timestamp', ''))
    
    # Statistics with styled cards
    st.markdown("### 📊 Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(all_messages))
    with col2:
        user_count = sum(1 for m in all_messages if m.get('role') == 'user')
        st.metric("User Messages", user_count)
    with col3:
        assistant_count = sum(1 for m in all_messages if m.get('role') == 'assistant')
        st.metric("AI Responses", assistant_count)
    with col4:
        tool_count = len(st.session_state.agent.memory.tool_calls) if st.session_state.agent and hasattr(st.session_state.agent, 'memory') else 0
        st.metric("Tool Calls", tool_count)
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        role_filter = st.selectbox("Filter by Role", ["All", "User", "Assistant"], key="history_role_filter")
    with col2:
        search_text = st.text_input("Search Messages", key="history_search", placeholder="Search...")
    with col3:
        limit = st.number_input("Show Last N", min_value=0, max_value=1000, value=min(50, len(all_messages)), key="history_limit")
    
    # Apply filters
    filtered = all_messages
    if role_filter != "All":
        filtered = [m for m in filtered if m.get('role', '').lower() == role_filter.lower()]
    if search_text:
        filtered = [m for m in filtered if search_text.lower() in m.get('content', '').lower()]
    if limit > 0:
        filtered = filtered[-limit:]
    
    st.markdown(f"**Showing {len(filtered)} of {len(all_messages)} messages**")
    
    # Export buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if filtered:
            df = pd.DataFrame(filtered)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    with col2:
        if filtered:
            import json
            json_data = json.dumps(filtered, indent=2, default=str)
            st.download_button(
                "📥 Export JSON",
                json_data,
                f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.agent and hasattr(st.session_state.agent, 'memory'):
                st.session_state.agent.memory.clear()
            st.success("Cleared!")
            st.rerun()
    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Display messages
    if filtered:
        for msg in filtered:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(timestamp)
            except:
                time_str = str(timestamp)
            
            if role == 'user':
                with st.chat_message("user"):
                    st.write(content)
                    st.caption(f"📅 {time_str}")
            elif role == 'assistant':
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(content)
                    st.caption(f"📅 {time_str}")
    else:
        st.info("No messages found. Start chatting to see history!")

# -----------------------
# Settings Page
# -----------------------

def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    st.caption("Configure AI models, data sources, and display preferences")
    
    # AI Model Configuration with styled container
    st.markdown("### 🤖 AI Model Configuration")
    st.markdown("""
    <div class="summary-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Choose your AI provider</h4>
    </div>
    """, unsafe_allow_html=True)
    
    has_openai_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY'))
    default_to_hf = not has_openai_key
    
    use_hf = st.checkbox(
        "Use Hugging Face (FREE!)",
        value=st.session_state.get('use_huggingface', default_to_hf)
    )
    st.session_state['use_huggingface'] = use_hf
    
    if use_hf:
        st.markdown("#### Hugging Face Models")
        hf_model = st.selectbox(
            "Select Model",
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
        st.info("💡 Free API. Get token for higher limits: https://huggingface.co/settings/tokens")
    else:
        st.markdown("#### OpenAI Models")
        model = st.selectbox(
            "Select Model",
            ["gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state['model'] = model
        if not has_openai_key:
            st.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    st.markdown("---")
    
    # Data Configuration with styled container
    st.markdown("### 📁 Data Configuration")
    st.markdown("""
    <div class="summary-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Manage your data source</h4>
    </div>
    """, unsafe_allow_html=True)
    
    data_path = st.text_input(
        "Data File Path",
        value=st.session_state.get('data_path', 'data/cpg_sales_data.parquet')
    )
    st.session_state['data_path'] = data_path
    
    if st.button("🔄 Reload Data", type="primary", use_container_width=True):
        if os.path.exists(data_path):
            try:
                data = load_cpg_data(data_path)
                metadata = get_data_summary(data)
                st.session_state.data = data
                st.session_state.metadata = metadata
                st.session_state.data_loaded = True
                
                if st.session_state.agent:
                    st.session_state.agent.load_data(data_path)
                
                st.success("✅ Data reloaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.error(f"❌ File not found: {data_path}")
    
    st.markdown("---")
    
    # Display Settings with styled container
    st.markdown("### 🎨 Display Settings")
    st.markdown("""
    <div class="summary-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Customize your view</h4>
    </div>
    """, unsafe_allow_html=True)
    
    chart_height = st.slider("Chart Height (pixels)", 300, 800, st.session_state.chart_height, 50)
    st.session_state.chart_height = chart_height
    st.caption(f"Current chart height: {chart_height}px")

# -----------------------
# Main App
# -----------------------

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Get current page from session state
    page = st.session_state.get('current_page', "🏠 Home")
    
    if page == "🏠 Home":
        render_home()
    elif page == "💬 AI Chat":
        render_chat()
    elif page == "📈 Analytics":
        render_analytics()
    elif page == "📊 Dashboard":
        render_dashboard()
    elif page == "🕘 Chat History":
        render_chat_history()
    elif page == "⚙️ Settings":
        render_settings()

if __name__ == "__main__":
    main()
