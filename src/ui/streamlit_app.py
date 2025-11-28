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
import json
from io import BytesIO

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
        st.markdown("# CPG Decision Support")
        st.markdown("---")
    
        
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"
        
        # Navigation pages - Enhanced with modern analytics features
        pages = {
            "Overview": "Overview",
            "AI Assistant": "AI Assistant",
            "Business Insights": "Business Insights",
            "Data Export": "Data Export",
            "Data Comparison": "Data Comparison",
            "Forecasting": "Forecasting",
            "Custom Reports": "Custom Reports",
            "Data Quality": "Data Quality",
            "Performance Benchmarking": "Performance Benchmarking",
            "Alert Management": "Alert Management",
            "Scenario Planning": "Scenario Planning",
            "KPI Dashboard": "KPI Dashboard",
            "A/B Testing": "A/B Testing",
            "Data Profiling": "Data Profiling"
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
    st.markdown("## Overview")
    st.markdown("""
    **Welcome to CPG Decision Support Platform**
    
    This is your central hub for data-driven decision making. Load your sales data to get started with comprehensive analytics, 
    AI-powered insights, and advanced business intelligence tools.
    
    **Key Features:**
    - 📊 **Data Loading & Management**: Upload and manage your CPG sales data
    - 💬 **AI Assistant**: Get intelligent answers about your data using natural language
    - 📈 **Analytics Tools**: Access 12+ specialized analytics features
    - 🔍 **Real-time Insights**: Monitor performance and detect anomalies
    """)
    
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
            ("💬", "Go to AI Assistant", "AI Assistant"),
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
                st.session_state.current_page = "AI Assistant"
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
                    st.session_state.current_page = "AI Assistant"
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
                    st.session_state.current_page = "AI Assistant"
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
    st.markdown("## AI Assistant")
    st.markdown("""
    **Intelligent Data Analysis Through Natural Language**
    
    Interact with your sales data using conversational AI. Ask questions in plain English and get comprehensive 
    analysis, insights, and recommendations powered by advanced language models.
    
    **Capabilities:**
    - 📊 **Data Analysis**: Ask about trends, patterns, and performance metrics
    - 🔍 **Anomaly Detection**: Identify unusual patterns in your sales data
    - 📈 **Scenario Simulation**: Explore "what-if" scenarios for pricing and promotions
    - 💡 **Strategic Recommendations**: Receive actionable business insights
    - 📋 **Report Generation**: Get detailed analysis reports and summaries
    
    **Example Questions:**
    - "What are the sales trends for the last quarter?"
    - "Compare performance across different stores"
    - "What would happen if we run a 15% discount promotion?"
    - "Detect any anomalies in sales data"
    """)
    
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
# Removed: Analytics Page (copied feature)
# -----------------------

# -----------------------
# Removed: Dashboard Page (copied feature)
# -----------------------

# -----------------------
# Removed: Chat History Page (copied feature)
# -----------------------

# -----------------------
# Removed: Settings Page (copied feature)
# -----------------------

# All copied features (Analytics, Dashboard, Chat History, Settings) have been removed
# Core features: Home, AI Chat
# New innovative feature: Smart Insights (auto-generates actionable business insights)

# -----------------------
# Smart Insights Page (Innovative Feature)
# -----------------------

def render_smart_insights():
    """Render Smart Insights page that auto-generates actionable business insights."""
    st.markdown("## Business Insights")
    st.markdown("""
    **Automated Intelligence for Strategic Decision Making**
    
    Generate comprehensive business insights automatically from your sales data. This feature uses AI to analyze 
    your data and provide key insights, opportunities, risks, and actionable recommendations.
    
    **What You Get:**
    - 🎯 **Key Business Insights**: Top 3 critical insights from your data
    - 📈 **Growth Opportunities**: Identified areas for revenue and performance improvement
    - ⚠️ **Risk Assessment**: Potential concerns and areas requiring attention
    - 💡 **Actionable Recommendations**: Specific steps to optimize your business
    
    **Use Cases:**
    - Executive reporting and strategic planning
    - Quick data assessment before meetings
    - Identifying hidden patterns and opportunities
    - Getting AI-powered recommendations for business decisions
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    # Initialize agent if needed
    if not st.session_state.agent:
        agent, _ = initialize_agent()
        if agent:
            st.session_state.agent = agent
    
    if not st.session_state.agent:
        st.error("❌ Agent not initialized. Please check settings.")
        return
    
    st.markdown("### 🎯 Generate Insights")
    st.markdown("Click the button below to automatically generate comprehensive business insights from your data.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Smart Insights", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing data and generating insights..."):
                try:
                    # Use agent to generate comprehensive insights
                    insights_query = """
                    Analyze the sales data and provide:
                    1. Top 3 key business insights
                    2. Critical opportunities for growth
                    3. Potential risks or concerns
                    4. Actionable recommendations
                    
                    Be specific with numbers and percentages.
                    """
                    
                    result = st.session_state.agent.run(insights_query, generate_memo=False)
                    insight_text = result.get('response', 'No insights generated.')
                    
                    st.session_state['smart_insights'] = insight_text
                    st.session_state['insights_generated'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating insights: {e}")
    
    st.markdown("---")
    
    # Display generated insights
    if st.session_state.get('insights_generated') and st.session_state.get('smart_insights'):
        st.markdown("### 📊 Generated Insights")
        
        insights = st.session_state['smart_insights']
        
        # Display in a nice formatted way
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h3 style="color: white; margin-bottom: 1rem;">💡 AI-Generated Business Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(insights)
        
        # Quick action buttons
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📋 Copy Insights", use_container_width=True, key="copy_insights"):
                st.code(insights, language=None)
                st.success("Insights copied! (Use Ctrl+C)")
        
        with action_col2:
            if st.button("🔄 Regenerate", use_container_width=True, key="regen_insights"):
                st.session_state['insights_generated'] = False
                st.session_state['smart_insights'] = None
                st.rerun()
        
        with action_col3:
            if st.button("💬 Ask Follow-up", use_container_width=True, key="followup_insights"):
                st.session_state.current_page = "AI Assistant"
                st.rerun()
    else:
        st.info("👆 Click 'Generate Smart Insights' to get started!")
        
        # Show preview of what will be analyzed
        if st.session_state.data_loaded and st.session_state.metadata:
            st.markdown("### 📈 Data Overview")
            md = st.session_state.metadata
            preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
            
            with preview_col1:
                st.metric("Total Records", f"{md.get('rows', 0):,}")
            with preview_col2:
                st.metric("Stores", md.get('stores', 'N/A'))
            with preview_col3:
                st.metric("SKUs", md.get('skus', 'N/A'))
            with preview_col4:
                if md.get('total_revenue'):
                    st.metric("Total Revenue", f"${md['total_revenue']:,.0f}")

# -----------------------
# Data Export Page
# -----------------------

def render_data_export():
    """Render data export page with multiple format options."""
    st.markdown("## Data Export")
    st.markdown("""
    **Comprehensive Data Export and Sharing**
    
    Export your data and analysis results in multiple formats for reporting, sharing, or further analysis in other tools.
    
    **Export Options:**
    - 📄 **Raw Data Export**: Export filtered or complete datasets in CSV, Excel, or JSON formats
    - 📈 **Analysis Results**: Export analysis results and insights as JSON for programmatic access
    - 📋 **Custom Aggregations**: Create and export aggregated data with custom grouping and functions
    
    **Features:**
    - **Flexible Filtering**: Filter by date range, category, or specific columns before export
    - **Multiple Formats**: CSV (universal), Excel (formatted), JSON (structured data)
    - **Custom Aggregations**: Group and aggregate data by any dimension (date, category, store, region, SKU)
    - **Analysis Export**: Save analysis results for documentation or integration with other systems
    
    **Use Cases:**
    - Creating reports for stakeholders
    - Sharing data with team members
    - Importing into Excel, Power BI, or other analytics tools
    - Archiving analysis results
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### 📥 Export Options")
    
    export_tab1, export_tab2, export_tab3 = st.tabs(["📄 Raw Data", "📈 Analysis Results", "📋 Custom Export"])
    
    with export_tab1:
        st.markdown("#### Export Raw Data")
        
        # Filters for export
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                min_date = data['date'].min().date()
                max_date = data['date'].max().date()
                date_range = st.date_input("Date Range", value=(min_date, max_date), key="export_date")
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    data = data[(data['date'].dt.date >= date_range[0]) & (data['date'].dt.date <= date_range[1])]
        
        with col2:
            if 'category' in data.columns:
                categories = ['All'] + sorted(data['category'].unique().tolist())
                selected_cat = st.selectbox("Category", categories, key="export_category")
                if selected_cat != 'All':
                    data = data[data['category'] == selected_cat]
        
        with col3:
            columns_to_export = st.multiselect(
                "Select Columns",
                options=data.columns.tolist(),
                default=data.columns.tolist(),
                key="export_columns"
            )
            if columns_to_export:
                data = data[columns_to_export]
        
        st.markdown(f"**Rows to export:** {len(data):,}")
        st.dataframe(data.head(100), use_container_width=True)
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = data.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"cpg_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            # Excel export (requires openpyxl)
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Data')
                excel_data = output.getvalue()
                st.download_button(
                    "📊 Download Excel",
                    excel_data,
                    f"cpg_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
        with col3:
            # JSON export
            json_data = data.to_json(orient='records', date_format='iso')
            st.download_button(
                "📋 Download JSON",
                json_data,
                f"cpg_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
    
    with export_tab2:
        st.markdown("#### Export Analysis Results")
        
        if st.session_state.get('analysis_results'):
            st.success("✅ Analysis results available for export")
            
            # Export all analysis results
            results_json = json.dumps(st.session_state.analysis_results, indent=2, default=str)
            st.download_button(
                "📥 Download Analysis Results (JSON)",
                results_json,
                f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
            
            # Show summary
            with st.expander("View Analysis Results Summary"):
                st.json(st.session_state.analysis_results)
        else:
            st.info("💡 Run analyses in AI Chat or Smart Insights to generate exportable results")
    
    with export_tab3:
        st.markdown("#### Custom Export Builder")
        
        st.markdown("Create a custom export with aggregated data")
        
        agg_col1, agg_col2 = st.columns(2)
        with agg_col1:
            group_by = st.multiselect(
                "Group By",
                options=['date', 'category', 'store_id', 'store_region', 'sku_id'],
                key="export_groupby"
            )
        with agg_col2:
            agg_func = st.selectbox(
                "Aggregation",
                ["sum", "mean", "count", "min", "max"],
                key="export_agg"
            )
        
        if group_by and st.button("Generate Custom Export"):
            try:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if 'revenue' in numeric_cols:
                    custom_data = data.groupby(group_by).agg({col: agg_func for col in numeric_cols}).reset_index()
                    st.dataframe(custom_data, use_container_width=True)
                    
                    csv_custom = custom_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download Custom Export",
                        csv_custom,
                        f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------
# Comparison Tool Page
# -----------------------

def render_comparison_tool():
    """Render comparison tool for comparing different data segments."""
    st.markdown("## Data Comparison")
    st.markdown("""
    **Side-by-Side Performance Analysis**
    
    Compare performance metrics across different dimensions to identify trends, patterns, and opportunities for optimization.
    
    **Comparison Types:**
    - 📅 **Time Period Comparison**: Compare performance between different time periods (e.g., Q1 vs Q2, This Month vs Last Month)
    - 📦 **Category Comparison**: Analyze performance across product categories
    - 🏪 **Store Comparison**: Compare performance between different store locations
    - 🗺️ **Regional Comparison**: Analyze geographic performance differences
    
    **Metrics Analyzed:**
    - Revenue changes and growth percentages
    - Units sold comparisons
    - Average price analysis
    - Visual charts and graphs for easy interpretation
    
    **Use Cases:**
    - Period-over-period performance analysis
    - Identifying best and worst performing categories/stores
    - Regional performance optimization
    - Understanding seasonal variations
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    comparison_type = st.radio(
        "Comparison Type",
        ["Time Period", "Category", "Store", "Region"],
        horizontal=True,
        key="comp_type"
    )
    
    if comparison_type == "Time Period":
        st.markdown("### 📅 Compare Time Periods")
        
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                period1_start = st.date_input("Period 1 Start", value=min_date, key="p1_start")
                period1_end = st.date_input("Period 1 End", value=min_date + pd.Timedelta(days=30), key="p1_end")
            with col2:
                period2_start = st.date_input("Period 2 Start", value=max_date - pd.Timedelta(days=30), key="p2_start")
                period2_end = st.date_input("Period 2 End", value=max_date, key="p2_end")
            
            p1_data = data[(data['date'].dt.date >= period1_start) & (data['date'].dt.date <= period1_end)]
            p2_data = data[(data['date'].dt.date >= period2_start) & (data['date'].dt.date <= period2_end)]
            
            # Comparison metrics
            st.markdown("### 📊 Comparison Metrics")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                p1_rev = p1_data['revenue'].sum() if 'revenue' in p1_data.columns else 0
                p2_rev = p2_data['revenue'].sum() if 'revenue' in p2_data.columns else 0
                change = ((p2_rev - p1_rev) / p1_rev * 100) if p1_rev > 0 else 0
                st.metric("Revenue", f"${p2_rev:,.0f}", f"{change:+.1f}%")
            
            with comp_col2:
                p1_units = p1_data['units_sold'].sum() if 'units_sold' in p1_data.columns else 0
                p2_units = p2_data['units_sold'].sum() if 'units_sold' in p2_data.columns else 0
                change = ((p2_units - p1_units) / p1_units * 100) if p1_units > 0 else 0
                st.metric("Units Sold", f"{p2_units:,.0f}", f"{change:+.1f}%")
            
            with comp_col3:
                p1_avg = p1_rev / p1_units if p1_units > 0 else 0
                p2_avg = p2_rev / p2_units if p2_units > 0 else 0
                change = ((p2_avg - p1_avg) / p1_avg * 100) if p1_avg > 0 else 0
                st.metric("Avg Price", f"${p2_avg:.2f}", f"{change:+.1f}%")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Period 1', x=['Revenue', 'Units'], y=[p1_rev/1000, p1_units/1000], marker_color='#667eea'))
            fig.add_trace(go.Bar(name='Period 2', x=['Revenue', 'Units'], y=[p2_rev/1000, p2_units/1000], marker_color='#48bb78'))
            fig.update_layout(title='Period Comparison', barmode='group', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_type == "Category":
        st.markdown("### 📦 Compare Categories")
        
        if 'category' in data.columns:
            categories = sorted(data['category'].unique().tolist())
            selected_cats = st.multiselect("Select Categories", categories, default=categories[:3] if len(categories) >= 3 else categories)
            
            if selected_cats:
                cat_comparison = data[data['category'].isin(selected_cats)].groupby('category').agg({
                    'revenue': 'sum',
                    'units_sold': 'sum'
                }).reset_index()
                
                st.dataframe(cat_comparison, use_container_width=True)
                
                fig = px.bar(cat_comparison, x='category', y='revenue', title='Revenue by Category')
                st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_type == "Store":
        st.markdown("### 🏪 Compare Stores")
        
        if 'store_id' in data.columns:
            stores = sorted(data['store_id'].unique().tolist())
            selected_stores = st.multiselect("Select Stores", stores, default=stores[:5] if len(stores) >= 5 else stores)
            
            if selected_stores:
                store_comparison = data[data['store_id'].isin(selected_stores)].groupby('store_id').agg({
                    'revenue': 'sum',
                    'units_sold': 'sum'
                }).reset_index()
                
                st.dataframe(store_comparison, use_container_width=True)
                
                fig = px.bar(store_comparison, x='store_id', y='revenue', title='Revenue by Store')
                st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_type == "Region":
        st.markdown("### 🗺️ Compare Regions")
        
        if 'store_region' in data.columns:
            region_comparison = data.groupby('store_region').agg({
                'revenue': 'sum',
                'units_sold': 'sum'
            }).reset_index()
            
            st.dataframe(region_comparison, use_container_width=True)
            
            fig = px.pie(region_comparison, values='revenue', names='store_region', title='Revenue Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Forecasting Page
# -----------------------

def render_forecasting():
    """Render forecasting page with predictive analytics."""
    st.markdown("## Forecasting")
    st.markdown("""
    **Predictive Analytics for Future Planning**
    
    Generate accurate sales forecasts and predictions based on your historical data to support inventory planning, 
    budgeting, and strategic decision-making.
    
    **Forecasting Methods:**
    - 📈 **Linear Trend**: Projects future values based on linear trend analysis
    - 📊 **Moving Average**: Uses historical averages to predict future performance
    - 📉 **Exponential Smoothing**: Applies weighted averages with more weight on recent data
    
    **Features:**
    - **Customizable Forecast Periods**: Forecast from 1 day to 365 days ahead
    - **Multiple Metrics**: Forecast revenue, units sold, or other key metrics
    - **Visual Forecasts**: Interactive charts showing historical data and future predictions
    - **Summary Statistics**: Average forecast, total forecast, and trend direction
    
    **Use Cases:**
    - Inventory planning and procurement
    - Budget forecasting and financial planning
    - Demand planning and capacity management
    - Setting sales targets and goals
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    if 'date' not in data.columns:
        st.error("Date column required for forecasting")
        return
    
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    st.markdown("### 📈 Forecast Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_metric = st.selectbox("Metric to Forecast", ["revenue", "units_sold"], key="forecast_metric")
    with col2:
        forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=365, value=30, key="forecast_periods")
    with col3:
        forecast_method = st.selectbox("Method", ["Linear Trend", "Moving Average", "Exponential Smoothing"], key="forecast_method")
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Aggregate data by date
                daily_data = data.groupby('date')[forecast_metric].sum().reset_index()
                daily_data = daily_data.set_index('date')
                
                # Simple forecasting methods
                if forecast_method == "Linear Trend":
                    from sklearn.linear_model import LinearRegression
                    
                    # Create time features
                    daily_data['days'] = (daily_data.index - daily_data.index.min()).days
                    X = daily_data[['days']].values
                    y = daily_data[forecast_metric].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Generate future dates
                    last_date = daily_data.index.max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                    future_days = [(d - daily_data.index.min()).days for d in future_dates]
                    future_X = np.array(future_days).reshape(-1, 1)
                    forecast_values = model.predict(future_X)
                    
                elif forecast_method == "Moving Average":
                    window = min(7, len(daily_data) // 4)
                    ma = daily_data[forecast_metric].rolling(window=window).mean().iloc[-1]
                    forecast_values = [ma] * forecast_periods
                    last_date = daily_data.index.max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                
                else:  # Exponential Smoothing
                    alpha = 0.3
                    forecast_values = []
                    last_value = daily_data[forecast_metric].iloc[-1]
                    last_date = daily_data.index.max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                    
                    for _ in range(forecast_periods):
                        forecast_values.append(last_value)
                        last_value = alpha * last_value + (1 - alpha) * last_value
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecast': forecast_values
                })
                
                st.session_state['forecast_data'] = forecast_df
                st.session_state['forecast_generated'] = True
                st.success("✅ Forecast generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
    
    if st.session_state.get('forecast_generated') and st.session_state.get('forecast_data') is not None:
        forecast_df = st.session_state['forecast_data']
        daily_data = data.groupby('date')[forecast_metric].sum().reset_index()
        
        st.markdown("### 📊 Forecast Visualization")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data[forecast_metric],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#667eea', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#f5576c', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{forecast_metric.replace("_", " ").title()} Forecast',
            xaxis_title='Date',
            yaxis_title=forecast_metric.replace('_', ' ').title(),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        st.markdown("### 📋 Forecast Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            avg_forecast = forecast_df['forecast'].mean()
            st.metric("Average Forecast", f"${avg_forecast:,.0f}" if forecast_metric == 'revenue' else f"{avg_forecast:,.0f}")
        with summary_col2:
            total_forecast = forecast_df['forecast'].sum()
            st.metric("Total Forecast", f"${total_forecast:,.0f}" if forecast_metric == 'revenue' else f"{total_forecast:,.0f}")
        with summary_col3:
            trend = "📈 Increasing" if forecast_df['forecast'].iloc[-1] > forecast_df['forecast'].iloc[0] else "📉 Decreasing"
            st.metric("Trend", trend)

# -----------------------
# Custom Reports Page
# -----------------------

def render_custom_reports():
    """Render custom report builder page."""
    st.markdown("## Custom Reports")
    st.markdown("""
    **Build Tailored Business Reports**
    
    Create customized reports with your preferred metrics, visualizations, and sections. Design reports that match 
    your specific business needs and reporting requirements.
    
    **Report Sections Available:**
    - 📊 **Summary Statistics**: Key metrics including total revenue, units, average price, and record counts
    - 💰 **Revenue Analysis**: Monthly revenue trends and patterns
    - 📦 **Category Breakdown**: Performance analysis by product category
    - 🏪 **Store Performance**: Individual store performance metrics
    - 📈 **Time Series Trends**: Historical trend analysis over time
    
    **Features:**
    - **Flexible Filtering**: Apply date range and category filters
    - **Customizable Sections**: Select only the sections you need
    - **Export Capability**: Download reports as text files for sharing
    - **Professional Formatting**: Clean, organized report layout
    
    **Use Cases:**
    - Executive dashboards and board presentations
    - Monthly/quarterly business reviews
    - Department-specific reporting
    - Regulatory compliance reporting
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### 🎨 Report Configuration")
    
    report_name = st.text_input("Report Name", value=f"Report_{datetime.now().strftime('%Y%m%d')}", key="report_name")
    
    # Report sections
    sections = st.multiselect(
        "Select Report Sections",
        ["Summary Statistics", "Revenue Analysis", "Category Breakdown", "Store Performance", "Time Series Trends"],
        default=["Summary Statistics", "Revenue Analysis"],
        key="report_sections"
    )
    
    # Filters
    with st.expander("🔍 Apply Filters"):
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                min_date = data['date'].min().date()
                max_date = data['date'].max().date()
                date_range = st.date_input("Date Range", value=(min_date, max_date), key="report_date")
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    data = data[(data['date'].dt.date >= date_range[0]) & (data['date'].dt.date <= date_range[1])]
        
        with filter_col2:
            if 'category' in data.columns:
                categories = ['All'] + sorted(data['category'].unique().tolist())
                selected_cat = st.selectbox("Category", categories, key="report_category")
                if selected_cat != 'All':
                    data = data[data['category'] == selected_cat]
    
    if st.button("📊 Generate Report", type="primary"):
        st.session_state['custom_report_data'] = data
        st.session_state['custom_report_sections'] = sections
        st.session_state['custom_report_name'] = report_name
        st.success("✅ Report generated!")
        st.rerun()
    
    if st.session_state.get('custom_report_data') is not None:
        report_data = st.session_state['custom_report_data']
        report_sections = st.session_state.get('custom_report_sections', [])
        report_name = st.session_state.get('custom_report_name', 'Report')
        
        st.markdown(f"### 📄 {report_name}")
        st.markdown("---")
        
        if "Summary Statistics" in report_sections:
            st.markdown("#### 📊 Summary Statistics")
            summary_stats = {
                'Total Revenue': f"${report_data['revenue'].sum():,.2f}" if 'revenue' in report_data.columns else "N/A",
                'Total Units': f"{report_data['units_sold'].sum():,.0f}" if 'units_sold' in report_data.columns else "N/A",
                'Average Price': f"${report_data['price'].mean():.2f}" if 'price' in report_data.columns else "N/A",
                'Total Records': len(report_data)
            }
            st.json(summary_stats)
        
        if "Revenue Analysis" in report_sections:
            st.markdown("#### 💰 Revenue Analysis")
            if 'date' in report_data.columns and 'revenue' in report_data.columns:
                revenue_trend = report_data.groupby(report_data['date'].dt.to_period('M'))['revenue'].sum()
                fig = px.line(x=revenue_trend.index.astype(str), y=revenue_trend.values, title='Monthly Revenue Trend')
                st.plotly_chart(fig, use_container_width=True)
        
        if "Category Breakdown" in report_sections:
            st.markdown("#### 📦 Category Breakdown")
            if 'category' in report_data.columns and 'revenue' in report_data.columns:
                cat_revenue = report_data.groupby('category')['revenue'].sum().sort_values(ascending=False)
                fig = px.bar(x=cat_revenue.index, y=cat_revenue.values, title='Revenue by Category')
                st.plotly_chart(fig, use_container_width=True)
        
        # Export report
        st.markdown("---")
        if st.button("📥 Export Report"):
            report_text = f"# {report_name}\n\n"
            report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_text += f"## Summary Statistics\n{summary_stats}\n\n"
            
            st.download_button(
                "📄 Download Report (TXT)",
                report_text,
                f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

# -----------------------
# Data Quality Page
# -----------------------

def render_data_quality():
    """Render data quality checker page."""
    st.markdown("## Data Quality")
    st.markdown("""
    **Data Validation and Quality Assessment**
    
    Ensure your data is clean, complete, and reliable before analysis. This tool performs comprehensive quality 
    checks and provides a quality score to help you understand data reliability.
    
    **Quality Checks Performed:**
    - ✅ **Missing Values**: Identifies columns with null or missing data
    - 🔍 **Duplicate Detection**: Finds duplicate rows in your dataset
    - ⚠️ **Data Type Validation**: Verifies correct data types (numeric, date, text)
    - 📊 **Value Validation**: Checks for negative values in numeric columns
    - 📅 **Date Consistency**: Validates date format and consistency
    
    **Quality Score:**
    - **0-100 Score**: Overall data quality rating
    - **Color-coded**: Green (80+), Yellow (60-79), Red (<60)
    - **Issue Reporting**: Detailed list of all quality issues found
    - **Completeness Report**: Column-by-column completeness percentage
    
    **Use Cases:**
    - Pre-analysis data validation
    - Data cleaning and preparation
    - Quality assurance for data imports
    - Identifying data entry errors
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    if st.button("🔍 Run Quality Check", type="primary"):
        quality_issues = []
        quality_score = 100
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0]
            quality_issues.append(f"⚠️ Missing values found in: {', '.join(missing_cols.index.tolist())}")
            quality_score -= (missing.sum() / len(data) * 100)
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"⚠️ {duplicates} duplicate rows found")
            quality_score -= (duplicates / len(data) * 10)
        
        # Check for negative values in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (data[col] < 0).any():
                quality_issues.append(f"⚠️ Negative values found in {col}")
                quality_score -= 5
        
        # Check date consistency
        if 'date' in data.columns:
            try:
                pd.to_datetime(data['date'])
            except:
                quality_issues.append("⚠️ Date column has invalid dates")
                quality_score -= 10
        
        # Check data types
        if 'revenue' in data.columns and not pd.api.types.is_numeric_dtype(data['revenue']):
            quality_issues.append("⚠️ Revenue column is not numeric")
            quality_score -= 15
        
        quality_score = max(0, quality_score)
        
        st.session_state['quality_issues'] = quality_issues
        st.session_state['quality_score'] = quality_score
        st.rerun()
    
    if st.session_state.get('quality_score') is not None:
        score = st.session_state['quality_score']
        issues = st.session_state.get('quality_issues', [])
        
        st.markdown("### 📊 Quality Score")
        
        # Score visualization
        score_color = "#48bb78" if score >= 80 else "#fbbf24" if score >= 60 else "#f5576c"
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: {score_color}; font-size: 4rem;">{score:.1f}/100</h1>
            <p style="font-size: 1.2rem;">Data Quality Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        if issues:
            st.markdown("### ⚠️ Issues Found")
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ No quality issues detected!")
        
        # Data completeness
        st.markdown("### 📋 Data Completeness")
        completeness_df = pd.DataFrame({
            'Column': data.columns,
            'Non-Null Count': [data[col].notna().sum() for col in data.columns],
            'Null Count': [data[col].isna().sum() for col in data.columns],
            'Completeness %': [f"{(data[col].notna().sum() / len(data) * 100):.1f}%" for col in data.columns]
        })
        st.dataframe(completeness_df, use_container_width=True)

# -----------------------
# Benchmarking Page
# -----------------------

def render_benchmarking():
    """Render performance benchmarking page."""
    st.markdown("## Performance Benchmarking")
    st.markdown("""
    **Target-Based Performance Evaluation**
    
    Set performance targets and compare your actual results against them. Track progress toward goals and identify 
    areas that need attention.
    
    **Benchmark Metrics:**
    - 💰 **Revenue Targets**: Set and track revenue goals
    - 📦 **Units Targets**: Monitor units sold against targets
    - 📈 **Growth Targets**: Track growth percentage goals
    
    **Features:**
    - **Performance Percentage**: See how close you are to targets (e.g., 95% of target)
    - **Visual Progress Bars**: Visual representation of target achievement
    - **Comparison Charts**: Side-by-side comparison of actual vs target performance
    - **Flexible Targets**: Set custom targets based on your business goals
    
    **Use Cases:**
    - KPI tracking and performance monitoring
    - Goal setting and achievement tracking
    - Budget vs actual analysis
    - Sales target monitoring
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### 🎯 Set Benchmarks")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        revenue_target = st.number_input("Revenue Target ($)", min_value=0.0, value=float(data['revenue'].sum()) if 'revenue' in data.columns else 0.0, key="bench_revenue")
    with col2:
        units_target = st.number_input("Units Target", min_value=0.0, value=float(data['units_sold'].sum()) if 'units_sold' in data.columns else 0.0, key="bench_units")
    with col3:
        growth_target = st.number_input("Growth Target (%)", min_value=0.0, value=10.0, key="bench_growth")
    
    if st.button("📊 Calculate Performance", type="primary"):
        actual_revenue = data['revenue'].sum() if 'revenue' in data.columns else 0
        actual_units = data['units_sold'].sum() if 'units_sold' in data.columns else 0
        
        revenue_performance = (actual_revenue / revenue_target * 100) if revenue_target > 0 else 0
        units_performance = (actual_units / units_target * 100) if units_target > 0 else 0
        
        st.session_state['benchmark_results'] = {
            'revenue_performance': revenue_performance,
            'units_performance': units_performance,
            'actual_revenue': actual_revenue,
            'actual_units': actual_units,
            'revenue_target': revenue_target,
            'units_target': units_target
        }
        st.rerun()
    
    if st.session_state.get('benchmark_results'):
        results = st.session_state['benchmark_results']
        
        st.markdown("### 📊 Performance vs Targets")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            rev_perf = results['revenue_performance']
            st.metric(
                "Revenue Performance",
                f"${results['actual_revenue']:,.0f}",
                f"{rev_perf:.1f}% of target"
            )
            st.progress(min(rev_perf / 100, 1.0))
        
        with perf_col2:
            units_perf = results['units_performance']
            st.metric(
                "Units Performance",
                f"{results['actual_units']:,.0f}",
                f"{units_perf:.1f}% of target"
            )
            st.progress(min(units_perf / 100, 1.0))
        
        # Visualization
        comparison_df = pd.DataFrame({
            'Metric': ['Revenue', 'Units'],
            'Actual': [results['actual_revenue']/1000, results['actual_units']/1000],
            'Target': [results['revenue_target']/1000, results['units_target']/1000]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Actual', x=comparison_df['Metric'], y=comparison_df['Actual'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Target', x=comparison_df['Metric'], y=comparison_df['Target'], marker_color='#48bb78'))
        fig.update_layout(title='Performance vs Targets', barmode='group', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Alert System Page
# -----------------------

def render_alert_system():
    """Render alert system for monitoring thresholds and anomalies."""
    st.markdown("## Alert Management")
    st.markdown("""
    **Automated Monitoring and Alerting**
    
    Set up automated alerts to monitor key metrics and get notified when thresholds are exceeded or anomalies are detected. 
    Stay informed about critical changes in your business performance.
    
    **Alert Types:**
    - 📊 **Threshold Alerts**: Monitor when metrics exceed or fall below specified thresholds
    - 🔍 **Anomaly Alerts**: Automatic alerts when unusual patterns are detected in your data
    - ⚠️ **Condition-Based**: Set custom conditions (>, <, >=, <=, ==) for any metric
    
    **Features:**
    - **Multiple Metrics**: Monitor revenue, units sold, price, or any numeric column
    - **Real-time Evaluation**: Alerts check current values against thresholds
    - **Active Alert Dashboard**: View and manage all configured alerts
    - **Alert Status**: See which alerts are triggered and which are within normal range
    
    **Use Cases:**
    - Revenue drop detection
    - Inventory threshold monitoring
    - Price change alerts
    - Anomaly detection notifications
    - Performance monitoring automation
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### ⚙️ Configure Alerts")
    
    alert_tab1, alert_tab2, alert_tab3 = st.tabs(["📊 Threshold Alerts", "🔍 Anomaly Alerts", "📋 Active Alerts"])
    
    with alert_tab1:
        st.markdown("#### Set Threshold Alerts")
        
        col1, col2 = st.columns(2)
        with col1:
            alert_metric = st.selectbox("Metric", ["revenue", "units_sold", "price"], key="alert_metric")
            alert_condition = st.selectbox("Condition", [">", "<", ">=", "<=", "=="], key="alert_condition")
        with col2:
            alert_threshold = st.number_input("Threshold Value", min_value=0.0, value=1000.0, key="alert_threshold")
            alert_name = st.text_input("Alert Name", value=f"{alert_metric} Alert", key="alert_name")
        
        if st.button("➕ Add Alert", type="primary"):
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
            
            alert = {
                'name': alert_name,
                'metric': alert_metric,
                'condition': alert_condition,
                'threshold': alert_threshold,
                'type': 'threshold',
                'active': True
            }
            st.session_state['alerts'].append(alert)
            st.success(f"✅ Alert '{alert_name}' added!")
            st.rerun()
    
    with alert_tab2:
        st.markdown("#### Anomaly Detection Alerts")
        
        # Check for anomalies button
        if st.button("🔍 Check for Anomalies", type="primary", key="check_anomalies"):
            try:
                with st.spinner("Analyzing data for anomalies..."):
                    from src.tools.anomaly_detection import get_anomaly_summary
                    anomaly_summary = get_anomaly_summary(data, metric='revenue')
                    
                    total_anomalies = (
                        anomaly_summary.get('statistical_anomalies', {}).get('count', 0) +
                        anomaly_summary.get('time_series_anomalies', {}).get('count', 0) +
                        anomaly_summary.get('multivariate_anomalies', {}).get('count', 0)
                    )
                    
                    # Store results in session state
                    st.session_state['anomaly_check_results'] = {
                        'total_anomalies': total_anomalies,
                        'anomaly_summary': anomaly_summary,
                        'checked': True
                    }
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error checking anomalies: {e}")
                st.session_state['anomaly_check_results'] = None
        
        # Display results if check has been performed
        if st.session_state.get('anomaly_check_results') is not None:
            results = st.session_state['anomaly_check_results']
            total_anomalies = results.get('total_anomalies', 0)
            
            if total_anomalies > 0:
                st.warning(f"⚠️ **{total_anomalies} anomalies detected!**")
                
                # Show breakdown
                anomaly_summary = results.get('anomaly_summary', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    stat_count = anomaly_summary.get('statistical_anomalies', {}).get('count', 0)
                    st.metric("Statistical", stat_count)
                with col2:
                    ts_count = anomaly_summary.get('time_series_anomalies', {}).get('count', 0)
                    st.metric("Time Series", ts_count)
                with col3:
                    mv_count = anomaly_summary.get('multivariate_anomalies', {}).get('count', 0)
                    st.metric("Multivariate", mv_count)
                
                # Alert name input
                anomaly_alert_name = st.text_input(
                    "Alert Name", 
                    value=f"Anomaly Alert ({total_anomalies} anomalies)",
                    key="anomaly_alert_name"
                )
                
                # Create alert button
                if st.button("➕ Create Anomaly Alert", type="primary", key="create_anomaly_alert"):
                    if 'alerts' not in st.session_state:
                        st.session_state['alerts'] = []
                    
                    # Check if alert with same name already exists
                    existing_names = [a.get('name', '') for a in st.session_state['alerts']]
                    if anomaly_alert_name in existing_names:
                        st.warning(f"⚠️ An alert with the name '{anomaly_alert_name}' already exists. Please use a different name.")
                    else:
                        alert = {
                            'name': anomaly_alert_name,
                            'type': 'anomaly',
                            'anomaly_count': total_anomalies,
                            'statistical_count': stat_count,
                            'time_series_count': ts_count,
                            'multivariate_count': mv_count,
                            'active': True,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state['alerts'].append(alert)
                        st.success(f"✅ **Anomaly alert '{anomaly_alert_name}' created successfully!**")
                        st.balloons()  # Celebration effect
                        st.rerun()
            else:
                st.success("✅ **No anomalies detected in your data.**")
                st.info("Your data appears to be normal. No alert needed.")
    
    with alert_tab3:
        st.markdown("#### Active Alerts")
        
        if 'alerts' in st.session_state and st.session_state['alerts']:
            active_alerts = [a for a in st.session_state['alerts'] if a.get('active', True)]
            
            if active_alerts:
                for idx, alert in enumerate(active_alerts):
                    with st.expander(f"🚨 {alert.get('name', 'Alert')} - {'Active' if alert.get('active') else 'Inactive'}"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if alert.get('type') == 'threshold':
                                st.write(f"**Metric:** {alert.get('metric')}")
                                st.write(f"**Condition:** {alert.get('condition')} {alert.get('threshold')}")
                                
                                # Check current value
                                if alert.get('metric') in data.columns:
                                    current_value = data[alert.get('metric')].sum() if alert.get('metric') in ['revenue', 'units_sold'] else data[alert.get('metric')].mean()
                                    st.write(f"**Current Value:** {current_value:,.2f}")
                                    
                                    # Evaluate condition
                                    condition_met = False
                                    if alert.get('condition') == '>':
                                        condition_met = current_value > alert.get('threshold')
                                    elif alert.get('condition') == '<':
                                        condition_met = current_value < alert.get('threshold')
                                    elif alert.get('condition') == '>=':
                                        condition_met = current_value >= alert.get('threshold')
                                    elif alert.get('condition') == '<=':
                                        condition_met = current_value <= alert.get('threshold')
                                    
                                    if condition_met:
                                        st.error("⚠️ ALERT TRIGGERED!")
                                    else:
                                        st.success("✅ Within normal range")
                            else:
                                st.write(f"**Type:** {alert.get('type')}")
                                st.write(f"**Anomaly Count:** {alert.get('anomaly_count', 0)}")
                                st.write(f"**Created:** {alert.get('timestamp', 'N/A')}")
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_{idx}"):
                                st.session_state['alerts'].pop(idx)
                                st.rerun()
            else:
                st.info("No active alerts configured")
        else:
            st.info("No alerts configured yet. Add alerts in the tabs above.")

# -----------------------
# What-If Scenario Builder Page
# -----------------------

def render_whatif_builder():
    """Render advanced what-if scenario builder."""
    st.markdown("## Scenario Planning")
    st.markdown("""
    **Advanced Multi-Variable Scenario Analysis**
    
    Build complex business scenarios by adjusting multiple variables simultaneously and see the combined impact on 
    your business metrics. Perfect for strategic planning and decision support.
    
    **Adjustable Variables:**
    - 💰 **Price Changes**: Adjust prices by percentage (-50% to +50%)
    - 🎁 **Promotion Settings**: Set discount percentage and coverage area
    - 📦 **Inventory Adjustments**: Modify inventory levels (-50% to +50%)
    - 📈 **Demand Shifts**: Simulate changes in market demand (-30% to +30%)
    
    **Analysis Provided:**
    - **Revenue Impact**: How scenario affects total revenue
    - **Units Impact**: Changes in units sold
    - **Price Analysis**: Average price changes
    - **Margin Impact**: Combined effect on profitability
    - **Visual Comparison**: Baseline vs scenario charts
    
    **Use Cases:**
    - Pricing strategy evaluation
    - Promotion planning and ROI analysis
    - Inventory optimization scenarios
    - Market condition simulation
    - Strategic planning and forecasting
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### 🎯 Scenario Configuration")
    
    scenario_name = st.text_input("Scenario Name", value=f"Scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="scenario_name")
    
    st.markdown("#### Variable Adjustments")
    
    var_col1, var_col2, var_col3 = st.columns(3)
    
    with var_col1:
        price_change = st.slider("Price Change (%)", min_value=-50.0, max_value=50.0, value=0.0, step=1.0, key="whatif_price")
        st.caption(f"New price: {data['price'].mean() * (1 + price_change/100):.2f}")
    
    with var_col2:
        promo_discount = st.slider("Promotion Discount (%)", min_value=0.0, max_value=50.0, value=0.0, step=5.0, key="whatif_promo")
        promo_coverage = st.slider("Promotion Coverage (%)", min_value=0.0, max_value=100.0, value=0.0, step=10.0, key="whatif_coverage")
    
    with var_col3:
        inventory_adjust = st.slider("Inventory Adjustment (%)", min_value=-50.0, max_value=50.0, value=0.0, step=5.0, key="whatif_inventory")
        demand_shift = st.slider("Demand Shift (%)", min_value=-30.0, max_value=30.0, value=0.0, step=1.0, key="whatif_demand")
    
    if st.button("🚀 Run Scenario", type="primary"):
        with st.spinner("Simulating scenario..."):
            try:
                # Create scenario data
                scenario_data = data.copy()
                
                # Apply price change
                scenario_data['new_price'] = scenario_data['price'] * (1 + price_change / 100)
                
                # Apply promotion
                if promo_coverage > 0:
                    num_promo_rows = int(len(scenario_data) * promo_coverage / 100)
                    promo_indices = np.random.choice(scenario_data.index, num_promo_rows, replace=False)
                    scenario_data.loc[promo_indices, 'new_price'] = scenario_data.loc[promo_indices, 'new_price'] * (1 - promo_discount / 100)
                
                # Apply demand shift (price elasticity effect)
                price_elasticity = -1.5
                demand_change = -price_elasticity * price_change / 100 + demand_shift / 100
                scenario_data['new_units'] = scenario_data['units_sold'] * (1 + demand_change)
                scenario_data['new_units'] = scenario_data['new_units'].clip(lower=0)
                
                # Apply inventory adjustment
                if 'inventory_level' in scenario_data.columns:
                    scenario_data['new_inventory'] = scenario_data['inventory_level'] * (1 + inventory_adjust / 100)
                
                # Calculate new revenue
                scenario_data['new_revenue'] = scenario_data['new_price'] * scenario_data['new_units']
                
                # Calculate metrics
                baseline_revenue = data['revenue'].sum()
                scenario_revenue = scenario_data['new_revenue'].sum()
                revenue_change = ((scenario_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0
                
                baseline_units = data['units_sold'].sum()
                scenario_units = scenario_data['new_units'].sum()
                units_change = ((scenario_units - baseline_units) / baseline_units * 100) if baseline_units > 0 else 0
                
                st.session_state['whatif_results'] = {
                    'scenario_name': scenario_name,
                    'baseline_revenue': baseline_revenue,
                    'scenario_revenue': scenario_revenue,
                    'revenue_change': revenue_change,
                    'baseline_units': baseline_units,
                    'scenario_units': scenario_units,
                    'units_change': units_change,
                    'scenario_data': scenario_data
                }
                st.success("✅ Scenario simulated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running scenario: {e}")
    
    if st.session_state.get('whatif_results'):
        results = st.session_state['whatif_results']
        
        st.markdown("### 📊 Scenario Results")
        
        st.markdown(f"#### {results['scenario_name']}")
        
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric("Revenue Change", f"${results['scenario_revenue']:,.0f}", f"{results['revenue_change']:+.1f}%")
        with result_col2:
            st.metric("Units Change", f"{results['scenario_units']:,.0f}", f"{results['units_change']:+.1f}%")
        with result_col3:
            avg_price = results['scenario_revenue'] / results['scenario_units'] if results['scenario_units'] > 0 else 0
            baseline_avg = results['baseline_revenue'] / results['baseline_units'] if results['baseline_units'] > 0 else 0
            price_change_pct = ((avg_price - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            st.metric("Avg Price", f"${avg_price:.2f}", f"{price_change_pct:+.1f}%")
        with result_col4:
            margin_change = results['revenue_change'] - results['units_change']
            st.metric("Margin Impact", f"{margin_change:+.1f}%", "vs baseline")
        
        # Visualization
        comparison_df = pd.DataFrame({
            'Metric': ['Revenue', 'Units'],
            'Baseline': [results['baseline_revenue']/1000, results['baseline_units']/1000],
            'Scenario': [results['scenario_revenue']/1000, results['scenario_units']/1000]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Baseline', x=comparison_df['Metric'], y=comparison_df['Baseline'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Scenario', x=comparison_df['Metric'], y=comparison_df['Scenario'], marker_color='#48bb78'))
        fig.update_layout(title='Baseline vs Scenario', barmode='group', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Performance Dashboard Page
# -----------------------

def render_performance_dashboard():
    """Render real-time performance dashboard with KPIs."""
    st.markdown("## KPI Dashboard")
    st.markdown("""
    **Real-Time Key Performance Indicator Monitoring**
    
    Monitor your business performance in real-time with a comprehensive dashboard of key metrics, trends, and insights. 
    Track KPIs and compare current performance with previous periods.
    
    **Key Metrics Tracked:**
    - 💰 **Total Revenue**: Current period revenue with growth percentage
    - 📦 **Total Units**: Units sold with period-over-period comparison
    - 💵 **Average Price**: Average transaction value
    - 🏪 **Active Stores**: Number of stores in operation
    
    **Dashboard Features:**
    - **Time Period Selection**: Analyze any custom date range
    - **Period Comparison**: Automatic comparison with previous period (YoY, MoM)
    - **Revenue Trends**: Daily revenue trend visualization
    - **Category Performance**: Top 5 categories by revenue
    - **Additional Metrics**: Promotion revenue, top regions, inventory levels
    
    **Use Cases:**
    - Daily/weekly/monthly performance monitoring
    - Executive dashboards
    - Quick performance health checks
    - Trend identification and analysis
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    # Time period selection
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            period_start = st.date_input("Start Date", value=min_date, key="dashboard_start")
        with col2:
            period_end = st.date_input("End Date", value=max_date, key="dashboard_end")
        
        data = data[(data['date'].dt.date >= period_start) & (data['date'].dt.date <= period_end)]
    
    st.markdown("### 📊 Key Performance Indicators")
    
    # Calculate KPIs
    total_revenue = data['revenue'].sum() if 'revenue' in data.columns else 0
    total_units = data['units_sold'].sum() if 'units_sold' in data.columns else 0
    avg_price = total_revenue / total_units if total_units > 0 else 0
    total_stores = data['store_id'].nunique() if 'store_id' in data.columns else 0
    total_skus = data['sku_id'].nunique() if 'sku_id' in data.columns else 0
    
    # Previous period comparison
    if 'date' in data.columns and len(data) > 0:
        period_days = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days
        prev_start = pd.to_datetime(period_start) - pd.Timedelta(days=period_days)
        prev_end = pd.to_datetime(period_start)
        
        prev_data = st.session_state.data.copy()
        prev_data['date'] = pd.to_datetime(prev_data['date'])
        prev_data = prev_data[(prev_data['date'].dt.date >= prev_start.date()) & (prev_data['date'].dt.date < prev_end.date())]
        
        prev_revenue = prev_data['revenue'].sum() if 'revenue' in prev_data.columns else 0
        prev_units = prev_data['units_sold'].sum() if 'units_sold' in prev_data.columns else 0
        
        revenue_growth = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        units_growth = ((total_units - prev_units) / prev_units * 100) if prev_units > 0 else 0
    else:
        revenue_growth = 0
        units_growth = 0
    
    # KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}", f"{revenue_growth:+.1f}%")
    with kpi_col2:
        st.metric("Total Units", f"{total_units:,.0f}", f"{units_growth:+.1f}%")
    with kpi_col3:
        st.metric("Avg Price", f"${avg_price:.2f}")
    with kpi_col4:
        st.metric("Active Stores", f"{total_stores}")
    
    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 📈 Revenue Trend")
        if 'date' in data.columns and 'revenue' in data.columns:
            daily_revenue = data.groupby(data['date'].dt.date)['revenue'].sum().reset_index()
            daily_revenue.columns = ['date', 'revenue']
            fig = px.line(daily_revenue, x='date', y='revenue', title='Daily Revenue Trend')
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### 📦 Category Performance")
        if 'category' in data.columns and 'revenue' in data.columns:
            cat_revenue = data.groupby('category')['revenue'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=cat_revenue.index, y=cat_revenue.values, title='Top 5 Categories by Revenue')
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    st.markdown("---")
    st.markdown("### 📊 Additional Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if 'promo_flag' in data.columns:
            promo_revenue = data[data['promo_flag'] == 1]['revenue'].sum() if len(data[data['promo_flag'] == 1]) > 0 else 0
            promo_pct = (promo_revenue / total_revenue * 100) if total_revenue > 0 else 0
            st.metric("Promotion Revenue", f"${promo_revenue:,.0f}", f"{promo_pct:.1f}% of total")
    
    with metric_col2:
        if 'store_region' in data.columns:
            top_region = data.groupby('store_region')['revenue'].sum().idxmax() if 'store_region' in data.columns else "N/A"
            st.metric("Top Region", top_region)
    
    with metric_col3:
        if 'inventory_level' in data.columns:
            avg_inventory = data['inventory_level'].mean()
            st.metric("Avg Inventory", f"{avg_inventory:,.0f}")

# -----------------------
# A/B Testing Analysis Page
# -----------------------

def render_ab_testing():
    """Render A/B testing analysis page."""
    st.markdown("## A/B Testing")
    st.markdown("""
    **Experimental Analysis for Strategy Optimization**
    
    Compare two different strategies, promotions, or scenarios side-by-side to determine which performs better. 
    Make data-driven decisions about pricing, promotions, and business strategies.
    
    **Test Configuration:**
    - **Control Group (Test A)**: Baseline strategy with current pricing and promotions
    - **Variant Group (Test B)**: Alternative strategy with different pricing/promotions
    - **Customizable Variables**: Price, promotion percentage, and coverage for each test
    
    **Analysis Provided:**
    - **Revenue Comparison**: Side-by-side revenue analysis
    - **Units Comparison**: Units sold comparison
    - **Average Price**: Price point analysis
    - **Winner Identification**: Automatic determination of best performing strategy
    - **Improvement Percentage**: Quantified improvement of winning strategy
    
    **Use Cases:**
    - Promotion effectiveness testing
    - Pricing strategy evaluation
    - Marketing campaign comparison
    - Product launch strategy testing
    - Optimization experiments
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    st.markdown("### 🎯 Test Configuration")
    
    test_tab1, test_tab2 = st.tabs(["📊 Configure Tests", "📈 Compare Results"])
    
    with test_tab1:
        st.markdown("#### Test A (Control)")
        
        test_a_col1, test_a_col2 = st.columns(2)
        with test_a_col1:
            test_a_name = st.text_input("Test A Name", value="Control", key="test_a_name")
            test_a_price = st.number_input("Price", min_value=0.01, value=float(data['price'].mean()) if 'price' in data.columns else 5.0, key="test_a_price")
        with test_a_col2:
            test_a_promo = st.slider("Promotion (%)", min_value=0.0, max_value=50.0, value=0.0, key="test_a_promo")
            test_a_coverage = st.slider("Coverage (%)", min_value=0.0, max_value=100.0, value=0.0, key="test_a_coverage")
        
        st.markdown("#### Test B (Variant)")
        
        test_b_col1, test_b_col2 = st.columns(2)
        with test_b_col1:
            test_b_name = st.text_input("Test B Name", value="Variant", key="test_b_name")
            test_b_price = st.number_input("Price", min_value=0.01, value=float(data['price'].mean() * 0.9) if 'price' in data.columns else 4.5, key="test_b_price")
        with test_b_col2:
            test_b_promo = st.slider("Promotion (%)", min_value=0.0, max_value=50.0, value=10.0, key="test_b_promo")
            test_b_coverage = st.slider("Coverage (%)", min_value=0.0, max_value=100.0, value=50.0, key="test_b_coverage")
        
        if st.button("🚀 Run A/B Test", type="primary"):
            with st.spinner("Running A/B test..."):
                try:
                    # Simulate Test A
                    test_a_data = data.copy()
                    test_a_data['test_price'] = test_a_price
                    if test_a_coverage > 0:
                        num_promo = int(len(test_a_data) * test_a_coverage / 100)
                        promo_indices = np.random.choice(test_a_data.index, num_promo, replace=False)
                        test_a_data.loc[promo_indices, 'test_price'] = test_a_data.loc[promo_indices, 'test_price'] * (1 - test_a_promo / 100)
                    
                    price_elasticity = -1.5
                    price_change_a = ((test_a_data['test_price'].mean() - data['price'].mean()) / data['price'].mean() * 100)
                    demand_change_a = -price_elasticity * price_change_a / 100
                    test_a_data['test_units'] = data['units_sold'] * (1 + demand_change_a)
                    test_a_data['test_revenue'] = test_a_data['test_price'] * test_a_data['test_units']
                    
                    # Simulate Test B
                    test_b_data = data.copy()
                    test_b_data['test_price'] = test_b_price
                    if test_b_coverage > 0:
                        num_promo = int(len(test_b_data) * test_b_coverage / 100)
                        promo_indices = np.random.choice(test_b_data.index, num_promo, replace=False)
                        test_b_data.loc[promo_indices, 'test_price'] = test_b_data.loc[promo_indices, 'test_price'] * (1 - test_b_promo / 100)
                    
                    price_change_b = ((test_b_data['test_price'].mean() - data['price'].mean()) / data['price'].mean() * 100)
                    demand_change_b = -price_elasticity * price_change_b / 100
                    test_b_data['test_units'] = data['units_sold'] * (1 + demand_change_b)
                    test_b_data['test_revenue'] = test_b_data['test_price'] * test_b_data['test_units']
                    
                    # Calculate results
                    baseline_revenue = data['revenue'].sum()
                    test_a_revenue = test_a_data['test_revenue'].sum()
                    test_b_revenue = test_b_data['test_revenue'].sum()
                    
                    test_a_change = ((test_a_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0
                    test_b_change = ((test_b_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0
                    
                    st.session_state['ab_test_results'] = {
                        'test_a': {
                            'name': test_a_name,
                            'revenue': test_a_revenue,
                            'change': test_a_change,
                            'units': test_a_data['test_units'].sum(),
                            'avg_price': test_a_data['test_price'].mean()
                        },
                        'test_b': {
                            'name': test_b_name,
                            'revenue': test_b_revenue,
                            'change': test_b_change,
                            'units': test_b_data['test_units'].sum(),
                            'avg_price': test_b_data['test_price'].mean()
                        },
                        'baseline': {
                            'revenue': baseline_revenue,
                            'units': data['units_sold'].sum()
                        }
                    }
                    st.success("✅ A/B test completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error running A/B test: {e}")
    
    with test_tab2:
        if st.session_state.get('ab_test_results'):
            results = st.session_state['ab_test_results']
            
            st.markdown("### 📊 A/B Test Results")
            
            # Comparison metrics
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown(f"#### {results['test_a']['name']}")
                st.metric("Revenue", f"${results['test_a']['revenue']:,.0f}", f"{results['test_a']['change']:+.1f}%")
                st.metric("Units", f"{results['test_a']['units']:,.0f}")
                st.metric("Avg Price", f"${results['test_a']['avg_price']:.2f}")
            
            with comp_col2:
                st.markdown(f"#### {results['test_b']['name']}")
                st.metric("Revenue", f"${results['test_b']['revenue']:,.0f}", f"{results['test_b']['change']:+.1f}%")
                st.metric("Units", f"{results['test_b']['units']:,.0f}")
                st.metric("Avg Price", f"${results['test_b']['avg_price']:.2f}")
            
            with comp_col3:
                st.markdown("#### Winner")
                winner = results['test_a']['name'] if results['test_a']['revenue'] > results['test_b']['revenue'] else results['test_b']['name']
                improvement = abs(results['test_a']['revenue'] - results['test_b']['revenue']) / min(results['test_a']['revenue'], results['test_b']['revenue']) * 100
                st.success(f"🏆 {winner}")
                st.metric("Improvement", f"{improvement:.1f}%")
            
            # Visualization
            comparison_df = pd.DataFrame({
                'Test': [results['test_a']['name'], results['test_b']['name']],
                'Revenue': [results['test_a']['revenue']/1000, results['test_b']['revenue']/1000],
                'Units': [results['test_a']['units']/1000, results['test_b']['units']/1000]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Revenue', x=comparison_df['Test'], y=comparison_df['Revenue'], marker_color='#667eea'))
            fig.add_trace(go.Bar(name='Units', x=comparison_df['Test'], y=comparison_df['Units'], marker_color='#48bb78', yaxis='y2'))
            fig.update_layout(
                title='A/B Test Comparison',
                barmode='group',
                template='plotly_white',
                yaxis=dict(title='Revenue (thousands)'),
                yaxis2=dict(title='Units (thousands)', overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run an A/B test in the 'Configure Tests' tab to see results here")

# -----------------------
# Data Profiling Page
# -----------------------

def render_data_profiling():
    """Render comprehensive data profiling page."""
    st.markdown("## Data Profiling")
    st.markdown("""
    **Comprehensive Statistical Analysis and Data Exploration**
    
    Perform deep statistical analysis of your dataset to understand data distributions, relationships, and characteristics. 
    Essential for data scientists and analysts who need detailed insights into their data structure.
    
    **Profiling Features:**
    - 📊 **Dataset Overview**: Total rows, columns, and memory usage
    - 📈 **Column Statistics**: For numeric columns - min, max, mean, median, std deviation, skewness, kurtosis
    - 📋 **Categorical Analysis**: For text/category columns - unique counts, top values, frequency distributions
    - 🔗 **Correlation Matrix**: Visual correlation analysis between all numeric columns
    - 📉 **Distribution Plots**: Histograms showing data distribution patterns
    
    **Statistical Measures:**
    - **Central Tendency**: Mean, median for numeric data
    - **Dispersion**: Standard deviation, range
    - **Shape**: Skewness (asymmetry), kurtosis (tail heaviness)
    - **Completeness**: Null counts and percentages per column
    
    **Use Cases:**
    - Data exploration and understanding
    - Statistical analysis and reporting
    - Data quality assessment
    - Feature engineering preparation
    - Identifying data relationships and patterns
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
        return
    
    data = st.session_state.data.copy()
    
    if st.button("🔍 Generate Profile", type="primary"):
        with st.spinner("Generating data profile..."):
            profile = {}
            
            # Basic statistics
            profile['shape'] = {'rows': len(data), 'columns': len(data.columns)}
            profile['memory_usage'] = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            
            # Column statistics
            profile['columns'] = {}
            for col in data.columns:
                col_info = {
                    'dtype': str(data[col].dtype),
                    'non_null_count': data[col].notna().sum(),
                    'null_count': data[col].isna().sum(),
                    'null_percentage': (data[col].isna().sum() / len(data) * 100)
                }
                
                if pd.api.types.is_numeric_dtype(data[col]):
                    col_info['min'] = float(data[col].min())
                    col_info['max'] = float(data[col].max())
                    col_info['mean'] = float(data[col].mean())
                    col_info['median'] = float(data[col].median())
                    col_info['std'] = float(data[col].std())
                    col_info['skewness'] = float(data[col].skew())
                    col_info['kurtosis'] = float(data[col].kurtosis())
                
                if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                    col_info['unique_count'] = data[col].nunique()
                    col_info['top_values'] = data[col].value_counts().head(5).to_dict()
                
                profile['columns'][col] = col_info
            
            st.session_state['data_profile'] = profile
            st.success("✅ Profile generated!")
            st.rerun()
    
    if st.session_state.get('data_profile'):
        profile = st.session_state['data_profile']
        
        st.markdown("### 📋 Overview")
        
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        with overview_col1:
            st.metric("Total Rows", f"{profile['shape']['rows']:,}")
        with overview_col2:
            st.metric("Total Columns", profile['shape']['columns'])
        with overview_col3:
            st.metric("Memory Usage", profile['memory_usage'])
        
        st.markdown("---")
        st.markdown("### 📊 Column Statistics")
        
        # Select column to view details
        selected_col = st.selectbox("Select Column", data.columns.tolist(), key="profile_column")
        
        if selected_col:
            col_info = profile['columns'].get(selected_col, {})
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("#### Basic Info")
                st.json({
                    'Data Type': col_info.get('dtype'),
                    'Non-Null Count': col_info.get('non_null_count'),
                    'Null Count': col_info.get('null_count'),
                    'Null Percentage': f"{col_info.get('null_percentage', 0):.2f}%"
                })
            
            with detail_col2:
                if pd.api.types.is_numeric_dtype(data[selected_col]):
                    st.markdown("#### Statistical Measures")
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("Mean", f"{col_info.get('mean', 0):.2f}")
                        st.metric("Median", f"{col_info.get('median', 0):.2f}")
                        st.metric("Std Dev", f"{col_info.get('std', 0):.2f}")
                    with stat_col2:
                        st.metric("Min", f"{col_info.get('min', 0):.2f}")
                        st.metric("Max", f"{col_info.get('max', 0):.2f}")
                        st.metric("Skewness", f"{col_info.get('skewness', 0):.2f}")
                else:
                    st.markdown("#### Categorical Info")
                    st.metric("Unique Values", col_info.get('unique_count', 0))
                    if col_info.get('top_values'):
                        st.markdown("**Top Values:**")
                        for val, count in list(col_info.get('top_values', {}).items())[:5]:
                            st.write(f"- {val}: {count}")
            
            # Distribution plot for numeric columns
            if pd.api.types.is_numeric_dtype(data[selected_col]):
                st.markdown("#### Distribution")
                fig = px.histogram(data, x=selected_col, title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for numeric columns
        st.markdown("---")
        st.markdown("### 🔗 Correlation Matrix")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")

# -----------------------
# Main App
# -----------------------

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Get current page from session state
    page = st.session_state.get('current_page', "Overview")
    
    if page == "Overview":
        render_home()
    elif page == "AI Assistant":
        render_chat()
    elif page == "Business Insights":
        render_smart_insights()
    elif page == "Data Export":
        render_data_export()
    elif page == "Data Comparison":
        render_comparison_tool()
    elif page == "Forecasting":
        render_forecasting()
    elif page == "Custom Reports":
        render_custom_reports()
    elif page == "Data Quality":
        render_data_quality()
    elif page == "Performance Benchmarking":
        render_benchmarking()
    elif page == "Alert Management":
        render_alert_system()
    elif page == "Scenario Planning":
        render_whatif_builder()
    elif page == "KPI Dashboard":
        render_performance_dashboard()
    elif page == "A/B Testing":
        render_ab_testing()
    elif page == "Data Profiling":
        render_data_profiling()
    # Removed routes: Analytics, Dashboard, Chat History, Settings (copied features)

if __name__ == "__main__":
    main()
