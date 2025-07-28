# ui/enhanced_streamlit_app.py - Enhanced InsightForge AI with Fallback Capabilities

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import InsightForge AI components
try:
    from app.enhanced_rag_pipeline import (
        EnhancedInsightForgeRAGPipeline,
        create_enhanced_rag_with_openai,
        create_enhanced_rag_with_anthropic,
        create_enhanced_rag_with_local_llm
    )
    from app.prompt_engineering import InsightForgePromptEngineering
    from app.memory import InsightForgeMemorySystem, ConversationContext
    from evaluation.improved_evaluator import create_improved_evaluator
    from utils.visualizations import InsightForgeVisualizer
    from loaders.data_loader import load_and_process_sales_data
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all InsightForge AI components are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="InsightForge AI Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB, #A23B72, #F18F01);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fallback-indicator {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .internal-source {
        background: #2E86AB;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
    .external-source {
        background: #FF6B6B;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
    .confidence-high {
        background: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
    }
    .confidence-medium {
        background: #FF9800;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
    }
    .confidence-low {
        background: #F44336;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background: #F0F7FF;
        border-left: 4px solid #2E86AB;
    }
    .enhanced-assistant-message {
        background: linear-gradient(135deg, #F0F7FF, #FFF8E1);
        border-left: 4px solid #FF6B6B;
        border-top: 2px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    
    if 'enhanced_initialized' not in st.session_state:
        st.session_state.enhanced_initialized = False
        st.session_state.enhanced_rag_pipeline = None
        st.session_state.prompt_engine = None
        st.session_state.memory_system = None
        st.session_state.improved_evaluator = None
        st.session_state.visualizer = None
        st.session_state.chat_history = []
        st.session_state.evaluation_results = []
        st.session_state.sales_data = None
        st.session_state.system_performance = {}
        st.session_state.current_session_id = None
        st.session_state.llm_configured = False
        st.session_state.api_key_input = ""
        st.session_state.tavily_api_key = ""
        st.session_state.fallback_enabled = True
        st.session_state.fallback_threshold = 0.3
        st.session_state.fallback_stats = {"total_queries": 0, "fallback_used": 0}

def check_environment_variables():
    """Check for API keys in environment variables."""
    
    env_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'tavily': os.getenv('TAVILY_API_KEY'),
    }
    
    return env_keys

def configure_enhanced_llm_provider(provider: str, api_key: str = None, tavily_key: str = None):
    """Configure the selected LLM provider with enhanced capabilities."""
    
    try:
        kb_path = os.path.join(project_root, "data", "knowledge_base")
        
        if provider == "OpenAI GPT":
            if not api_key:
                st.error("OpenAI API key is required")
                return None
            
            return create_enhanced_rag_with_openai(
                api_key=api_key,
                knowledge_base_path=kb_path,
                tavily_api_key=tavily_key,
                enable_fallback=st.session_state.fallback_enabled
            )
            
        elif provider == "Anthropic Claude":
            if not api_key:
                st.error("Anthropic API key is required")
                return None
            
            return create_enhanced_rag_with_anthropic(
                api_key=api_key,
                knowledge_base_path=kb_path,
                tavily_api_key=tavily_key,
                enable_fallback=st.session_state.fallback_enabled
            )
        
        elif provider == "Local Model":
            return create_enhanced_rag_with_local_llm(
                model_path="llama2",
                knowledge_base_path=kb_path,
                tavily_api_key=tavily_key,
                enable_fallback=st.session_state.fallback_enabled
            )
        
        else:  # Mock LLM
            return EnhancedInsightForgeRAGPipeline(
                knowledge_base_path=kb_path,
                llm=None,
                tavily_api_key=tavily_key,
                enable_fallback=st.session_state.fallback_enabled
            )
            
    except Exception as e:
        st.error(f"Error configuring {provider}: {e}")
        return None

@st.cache_resource
def load_enhanced_insightforge_systems(_llm_pipeline=None):
    """Load and cache enhanced InsightForge AI systems."""
    
    try:
        systems = {
            'enhanced_rag_pipeline': _llm_pipeline,
            'prompt_engine': InsightForgePromptEngineering(),
            'memory_system': InsightForgeMemorySystem(
                memory_type="buffer_window",
                window_size=10,
                persist_memory=True
            ),
            'improved_evaluator': create_improved_evaluator(),
            'visualizer': InsightForgeVisualizer()
        }
        
        return systems
        
    except Exception as e:
        st.error(f"Error loading enhanced InsightForge systems: {e}")
        return None

@st.cache_data
def load_sales_data():
    """Load and cache sales data."""
    
    try:
        sales_data_path = os.path.join(project_root, "data", "sales_data.csv")
        if os.path.exists(sales_data_path):
            return pd.read_csv(sales_data_path)
        else:
            return create_sample_data()
            
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample sales data for demonstration."""
    
    np.random.seed(42)
    n_records = 1000
    
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-12-31')
    dates = pd.date_range(start_date, end_date, periods=n_records)
    
    return pd.DataFrame({
        'Date': dates,
        'Sales': np.random.lognormal(4.2, 0.8, n_records),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n_records, p=[0.3, 0.25, 0.3, 0.15]),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records, p=[0.2, 0.2, 0.25, 0.35]),
        'Customer_Age': np.clip(np.random.normal(35, 12, n_records), 18, 75).astype(int),
        'Customer_Gender': np.random.choice(['Male', 'Female'], n_records, p=[0.45, 0.55]),
        'Customer_Satisfaction': np.random.uniform(1, 5, n_records)
    })

def main_header():
    """Display enhanced main application header."""
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 InsightForge AI Enhanced</h1>
        <h3>Advanced Business Intelligence with External Knowledge Access</h3>
        <p>Ask any business question - I'll search your data, documents, and the web for comprehensive insights</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.6rem; border-radius: 15px; margin: 0.2rem;">🗃️ Internal Knowledge</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.6rem; border-radius: 15px; margin: 0.2rem;">🌐 Web Search</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.6rem; border-radius: 15px; margin: 0.2rem;">📚 Wikipedia</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def enhanced_sidebar_configuration():
    """Enhanced sidebar configuration with fallback settings."""
    
    st.sidebar.markdown("## ⚙️ Enhanced System Configuration")
    
    # System status
    with st.sidebar.expander("🔍 Enhanced System Status", expanded=True):
        if st.session_state.enhanced_initialized and st.session_state.llm_configured:
            st.markdown('✅ **All Systems Operational**', unsafe_allow_html=True)
            st.write("🤖 Enhanced RAG Pipeline: Ready")
            st.write("🧠 Memory System: Active") 
            st.write("📚 Internal Knowledge Base: Active")
            
            # Show fallback status
            if st.session_state.enhanced_rag_pipeline:
                fallback_status = st.session_state.enhanced_rag_pipeline.get_fallback_status()
                st.write("🌐 **Fallback Capabilities:**")
                if fallback_status['tavily_available']:
                    st.write("  ✅ Tavily Web Search")
                else:
                    st.write("  ❌ Tavily Web Search")
                if fallback_status['wikipedia_available']:
                    st.write("  ✅ Wikipedia")
                else:
                    st.write("  ❌ Wikipedia")
                if fallback_status['agent_available']:
                    st.write("  ✅ Intelligent Agent")
                else:
                    st.write("  ❌ Intelligent Agent")
        elif st.session_state.enhanced_initialized:
            st.markdown('⚠️ **LLM Configuration Needed**', unsafe_allow_html=True)
        else:
            st.markdown('🔄 **Initializing Enhanced Systems**', unsafe_allow_html=True)
    
    # LLM Configuration
    st.sidebar.markdown("## 🔧 LLM Configuration")
    
    llm_option = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ["Mock LLM (Testing)", "OpenAI GPT", "Anthropic Claude", "Local Model"],
        help="Select your preferred language model"
    )
    
    # Check environment variables
    env_keys = check_environment_variables()
    
    api_key = None
    if llm_option == "OpenAI GPT":
        if env_keys['openai']:
            st.sidebar.success("🔑 OpenAI API key found in environment")
            api_key = env_keys['openai']
        else:
            api_key = st.sidebar.text_input(
                "OpenAI API Key:",
                type="password",
                value=st.session_state.api_key_input,
                help="Enter your OpenAI API key for GPT integration"
            )
            if api_key:
                st.session_state.api_key_input = api_key
    
    elif llm_option == "Anthropic Claude":
        if env_keys['anthropic']:
            st.sidebar.success("🔑 Anthropic API key found in environment")
            api_key = env_keys['anthropic']
        else:
            api_key = st.sidebar.text_input(
                "Anthropic API Key:",
                type="password",
                value=st.session_state.api_key_input,
                help="Enter your Anthropic API key for Claude integration"
            )
            if api_key:
                st.session_state.api_key_input = api_key
    
    elif llm_option == "Local Model":
        st.sidebar.info("🖥️ Using local Ollama model")
        api_key = "local"
    
    else:  # Mock LLM
        st.sidebar.info("🧪 Using Mock LLM for testing")
        api_key = "mock"
    
    # Fallback Configuration
    st.sidebar.markdown("## 🌐 Fallback Configuration")
    
    # Tavily API Key
    tavily_key = None
    if env_keys['tavily']:
        st.sidebar.success("🔑 Tavily API key found in environment")
        tavily_key = env_keys['tavily']
    else:
        tavily_key = st.sidebar.text_input(
            "Tavily API Key (Optional):",
            type="password",
            value=st.session_state.tavily_api_key,
            help="Enter your Tavily API key for web search capabilities"
        )
        if tavily_key:
            st.session_state.tavily_api_key = tavily_key
    
    # Fallback settings
    st.session_state.fallback_enabled = st.sidebar.checkbox(
        "Enable External Fallback",
        value=st.session_state.fallback_enabled,
        help="Use external sources when internal knowledge is insufficient"
    )
    
    if st.session_state.fallback_enabled:
        st.session_state.fallback_threshold = st.sidebar.slider(
            "Fallback Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.fallback_threshold,
            step=0.1,
            help="Lower values trigger fallback more often"
        )
    
    # Configure Enhanced LLM button
    if st.sidebar.button("🚀 Configure Enhanced System", type="primary"):
        if llm_option in ["Mock LLM (Testing)", "Local Model"] or api_key:
            with st.spinner("Configuring enhanced system..."):
                enhanced_pipeline = configure_enhanced_llm_provider(llm_option, api_key, tavily_key)
                
                if enhanced_pipeline:
                    # Update fallback threshold if pipeline exists
                    if st.session_state.fallback_enabled:
                        enhanced_pipeline.set_fallback_threshold(st.session_state.fallback_threshold)
                    
                    # Clear cache and reload
                    st.cache_resource.clear()
                    
                    systems = load_enhanced_insightforge_systems(enhanced_pipeline)
                    
                    if systems:
                        st.session_state.enhanced_rag_pipeline = systems['enhanced_rag_pipeline']
                        st.session_state.prompt_engine = systems['prompt_engine']
                        st.session_state.memory_system = systems['memory_system']
                        st.session_state.improved_evaluator = systems['improved_evaluator']
                        st.session_state.visualizer = systems['visualizer']
                        st.session_state.llm_configured = True
                        
                        st.sidebar.success(f"✅ Enhanced {llm_option} configured!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Failed to configure enhanced system")
                else:
                    st.sidebar.error("❌ Failed to configure enhanced LLM")
        else:
            st.sidebar.error("Please provide an API key")
    
    # Test Fallback Tools
    if st.session_state.enhanced_rag_pipeline and st.sidebar.button("🧪 Test Fallback Tools"):
        with st.spinner("Testing fallback capabilities..."):
            test_results = st.session_state.enhanced_rag_pipeline.test_fallback_tools()
            
            st.sidebar.markdown("### 🧪 Test Results:")
            if test_results.get('tavily_result'):
                st.sidebar.success("✅ Tavily working")
            if test_results.get('wikipedia_result'):
                st.sidebar.success("✅ Wikipedia working")
            if test_results.get('agent_result'):
                st.sidebar.success("✅ Agent working")
    
    # Fallback Statistics
    if st.session_state.fallback_stats["total_queries"] > 0:
        st.sidebar.markdown("## 📊 Fallback Statistics")
        fallback_rate = (st.session_state.fallback_stats["fallback_used"] / 
                        st.session_state.fallback_stats["total_queries"] * 100)
        st.sidebar.metric("Fallback Usage Rate", f"{fallback_rate:.1f}%")
        st.sidebar.metric("Total Queries", st.session_state.fallback_stats["total_queries"])
        st.sidebar.metric("Fallback Used", st.session_state.fallback_stats["fallback_used"])



# Fix for enhanced_streamlit_app.py - Chat Display Issue

# Replace the chat display section in enhanced_streamlit_app.py with this:

def enhanced_chat_interface():
    """Enhanced chat interface with fallback indicators."""
    
    st.header("💬 Enhanced AI Assistant - Ask Anything!")
    
    # Display configuration warning if needed
    if not st.session_state.llm_configured:
        st.warning("⚠️ Please configure an LLM provider in the sidebar to get AI-powered responses with fallback capabilities.")
    
    # Chat container
    chat_container = st.container()
    
    # Display enhanced chat history with FIXED formatting
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                # User message - simple display
                st.markdown("**You:** " + message["content"])
                st.markdown("---")
            else:
                # Assistant message - clean display without HTML
                st.markdown("**InsightForge AI Enhanced:** " + message["content"])
                
                # Show fallback indicator if used
                if message.get("used_fallback"):
                    strategy = message.get("fallback_strategy", "external")
                    reason = message.get("fallback_reason", "Low confidence")
                    st.info(f"🌐 Enhanced with {strategy} search - {reason}")
                
                # Show sources in a clean way
                if message.get("sources"):
                    with st.expander("📚 Sources Used"):
                        internal_sources = []
                        external_sources = []
                        
                        for src in message["sources"]:
                            if "Sales Data" in src.get("source_type", "") or "PDF" in src.get("source_type", ""):
                                internal_sources.append(src.get("source_type", "Internal"))
                            else:
                                external_sources.append(src.get("source_type", "External"))
                        
                        if internal_sources:
                            st.write("📊 **Internal Sources:**", ", ".join(set(internal_sources)))
                        if external_sources:
                            st.write("🌐 **External Sources:**", ", ".join(set(external_sources)))
                
                # Show metrics in a clean way
                if message.get("response_time"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Response Time", f"{message.get('response_time', 0):.2f}s")
                    with col2:
                        accuracy = message.get("accuracy", 0)
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col3:
                        relevance = message.get("relevance", 0)
                        st.metric("Relevance", f"{relevance:.3f}")
                    with col4:
                        completeness = message.get("completeness", 0)
                        st.metric("Completeness", f"{completeness:.3f}")
                
                st.markdown("---")
    
    # Enhanced Quick Questions (keep this part the same)
    st.markdown("### 🚀 Try These Enhanced Questions")
    st.markdown("*The system searches internal data first, then web sources if needed*")
    
    # Create enhanced question categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Internal Business Questions**")
        internal_questions = [
            "What are our total sales figures?",
            "Which products perform best?",
            "How do regions compare?"
        ]
        for i, question in enumerate(internal_questions):
            if st.button(question, key=f"internal_q_{i}", use_container_width=True):
                process_enhanced_user_query(question)
    
    with col2:
        st.markdown("**🌐 Current/External Questions**")
        external_questions = [
            "What are the latest AI business trends?",
            "Current best practices in sales analytics?",
            "Recent developments in customer intelligence?"
        ]
        for i, question in enumerate(external_questions):
            if st.button(question, key=f"external_q_{i}", use_container_width=True):
                process_enhanced_user_query(question)
    
    with col3:
        st.markdown("**🔄 Hybrid Questions**")
        hybrid_questions = [
            "How can current AI innovations improve our sales?",
            "What strategic recommendations based on industry trends?",
            "How do our metrics compare to industry standards?"
        ]
        for i, question in enumerate(hybrid_questions):
            if st.button(question, key=f"hybrid_q_{i}", use_container_width=True):
                process_enhanced_user_query(question)
    
    # Enhanced chat input
    user_query = st.chat_input("Ask me anything - I'll search your data and the web for comprehensive insights...")
    
    if user_query:
        process_enhanced_user_query(user_query)



def process_enhanced_user_query(query: str):
    """Process user query through the enhanced InsightForge AI pipeline with fallback."""
    
    if not st.session_state.enhanced_initialized:
        st.error("Please wait for enhanced system initialization to complete.")
        return
    
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now()
    })
    
    # Update statistics
    st.session_state.fallback_stats["total_queries"] += 1
    
    # Show enhanced processing indicator
    with st.spinner("🧠 InsightForge AI Enhanced is analyzing your question across all knowledge sources..."):
        
        try:
            start_time = time.time()
            
            # Step 1: Get enhanced response from pipeline
            response = st.session_state.enhanced_rag_pipeline.query(query, return_sources=True)
            
            # Update fallback statistics
            if response.get("used_fallback"):
                st.session_state.fallback_stats["fallback_used"] += 1
            
            # Step 2: Enhance with memory context
            if st.session_state.memory_system:
                memory_context = st.session_state.memory_system.get_conversation_context()
                if memory_context.get('recent_topics'):
                    context_note = f"\n\n*Building on our previous discussion about {', '.join(memory_context['recent_topics'][-2:])}, this enhanced analysis provides additional insights.*"
                    response['answer'] += context_note
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Step 3: Evaluate response quality with improved evaluator
            eval_result = None
            try:
                if st.session_state.improved_evaluator:
                    eval_result = st.session_state.improved_evaluator.evaluate_response(
                        question=query,
                        predicted_answer=response.get('answer', ''),
                        reference_answer=None,
                        response_time=response_time,
                        sources_used=[src.get('source_type', '') for src in response.get('sources', [])]
                    )
                    
                    st.session_state.evaluation_results.append(eval_result)
                    
            except Exception as e:
                st.warning(f"Evaluation error: {e}")
            
            # Step 4: Add to memory
            if st.session_state.memory_system:
                try:
                    context_type = detect_context_type(query)
                    st.session_state.memory_system.add_conversation_turn(
                        question=query,
                        answer=response.get('answer', ''),
                        context_type=context_type,
                        sources_used=[src.get('source_type', '') for src in response.get('sources', [])],
                        key_insights=extract_key_insights(response.get('answer', ''))
                    )
                except Exception as e:
                    st.warning(f"Memory integration error: {e}")
            
            # Add enhanced assistant response to chat
            assistant_message = {
                "role": "assistant",
                "content": response.get('answer', 'I apologize, but I could not generate a response.'),
                "timestamp": datetime.now(),
                "response_time": response_time,
                "accuracy": eval_result.accuracy_score if eval_result else 0,
                "relevance": eval_result.relevance_score if eval_result else 0,
                "completeness": eval_result.completeness_score if eval_result else 0,
                "sources": response.get('sources', []),
                "used_fallback": response.get('used_fallback', False),
                "fallback_strategy": response.get('fallback_strategy', ''),
                "fallback_reason": response.get('fallback_reason', '')
            }
            
            st.session_state.chat_history.append(assistant_message)
            
            # Update system performance metrics
            update_enhanced_performance_metrics(response_time, eval_result, response.get('used_fallback', False))
            
        except Exception as e:
            error_message = f"I encountered an error while processing your enhanced query: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_message,
                "timestamp": datetime.now(),
                "used_fallback": False
            })
    
    # Refresh the page to show new messages
    st.rerun()

def detect_context_type(query: str) -> ConversationContext:
    """Detect conversation context type from query."""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['sales', 'revenue', 'performance']):
        return ConversationContext.SALES_ANALYSIS
    elif any(word in query_lower for word in ['customer', 'demographic', 'age', 'gender']):
        return ConversationContext.CUSTOMER_INSIGHTS
    elif any(word in query_lower for word in ['product', 'item']):
        return ConversationContext.PRODUCT_PERFORMANCE
    elif any(word in query_lower for word in ['region', 'area', 'geographic']):
        return ConversationContext.REGIONAL_ANALYSIS
    elif any(word in query_lower for word in ['ai', 'artificial intelligence', 'strategy', 'innovation']):
        return ConversationContext.STRATEGIC_PLANNING
    else:
        return ConversationContext.GENERAL_BI

def extract_key_insights(response: str) -> List[str]:
    """Extract key insights from response for memory storage."""
    
    insights = []
    
    # Look for monetary amounts
    import re
    money_pattern = r'\$[\d,]+(?:\.\d{2})?'
    money_matches = re.findall(money_pattern, response)
    if money_matches:
        insights.extend([f"Financial metric: {amount}" for amount in money_matches[:2]])
    
    # Look for percentages
    percent_pattern = r'\d+(?:\.\d+)?%'
    percent_matches = re.findall(percent_pattern, response)
    if percent_matches:
        insights.extend([f"Performance metric: {pct}" for pct in percent_matches[:2]])
    
    # Look for AI/strategy mentions
    if any(word in response.lower() for word in ['ai', 'artificial intelligence', 'strategy']):
        insights.append("Strategic AI insights")
    
    # Look for external/current information
    if any(word in response.lower() for word in ['current', 'recent', 'latest', 'trend']):
        insights.append("Current market intelligence")
    
    return insights[:4]

def update_enhanced_performance_metrics(response_time: float, eval_result: Any, used_fallback: bool):
    """Update enhanced system performance metrics."""
    
    if 'response_times' not in st.session_state.system_performance:
        st.session_state.system_performance['response_times'] = []
        st.session_state.system_performance['accuracy_scores'] = []
        st.session_state.system_performance['relevance_scores'] = []
        st.session_state.system_performance['completeness_scores'] = []
        st.session_state.system_performance['fallback_usage'] = []
        st.session_state.system_performance['total_queries'] = 0
    
    st.session_state.system_performance['response_times'].append(response_time)
    st.session_state.system_performance['fallback_usage'].append(used_fallback)
    st.session_state.system_performance['total_queries'] += 1
    
    if eval_result:
        st.session_state.system_performance['accuracy_scores'].append(eval_result.accuracy_score)
        st.session_state.system_performance['relevance_scores'].append(eval_result.relevance_score)
        st.session_state.system_performance['completeness_scores'].append(eval_result.completeness_score)

def enhanced_system_performance_tab():
    """Display enhanced system performance metrics with fallback analytics."""
    
    st.header("🧪 Enhanced System Performance & Fallback Analytics")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation data available yet. Start asking questions to see enhanced performance metrics!")
        return
    
    # Enhanced performance overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    results = st.session_state.evaluation_results
    
    if results:
        avg_accuracy = np.mean([r.accuracy_score for r in results])
        avg_relevance = np.mean([r.relevance_score for r in results])
        avg_completeness = np.mean([r.completeness_score for r in results])
        avg_response_time = np.mean([r.response_time for r in results])
        
        col1.metric("Avg Accuracy", f"{avg_accuracy:.3f}")
        col2.metric("Avg Relevance", f"{avg_relevance:.3f}")
        col3.metric("Avg Completeness", f"{avg_completeness:.3f}")
        col4.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        # Fallback metrics
        total_queries = st.session_state.fallback_stats["total_queries"]
        fallback_used = st.session_state.fallback_stats["fallback_used"]
        fallback_rate = (fallback_used / total_queries * 100) if total_queries > 0 else 0
        col5.metric("Fallback Rate", f"{fallback_rate:.1f}%")
        
        # Enhanced performance charts
        st.subheader("📈 Enhanced Performance Analytics")
        
        # Create performance comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics over time
            metrics_data = []
            for i, result in enumerate(results[-20:], 1):  # Last 20 results
                metrics_data.append({
                    'Query': f"Q{i}",
                    'Accuracy': result.accuracy_score,
                    'Relevance': result.relevance_score,
                    'Completeness': result.completeness_score,
                    'Business_Relevance': getattr(result, 'business_relevance', 0.7)
                })
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                fig = px.line(df, x='Query', y=['Accuracy', 'Relevance', 'Completeness', 'Business_Relevance'], 
                             title="Performance Metrics Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fallback usage visualization
            if st.session_state.system_performance.get('fallback_usage'):
                fallback_data = st.session_state.system_performance['fallback_usage']
                fallback_df = pd.DataFrame({
                    'Query': range(1, len(fallback_data) + 1),
                    'Used_Fallback': ['Yes' if x else 'No' for x in fallback_data]
                })
                
                fig = px.histogram(fallback_df, x='Used_Fallback', 
                                 title="Fallback Usage Distribution",
                                 color='Used_Fallback',
                                 color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed evaluation breakdown
        st.subheader("📝 Recent Enhanced Evaluations")
        for i, result in enumerate(results[-5:], 1):  # Last 5 results
            with st.expander(f"Enhanced Query {len(results)-5+i}: {result.question[:60]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Question**: {result.question}")
                    st.write(f"**Answer Preview**: {result.predicted_answer[:300]}...")
                    st.write(f"**Evaluation Notes**: {result.evaluation_notes}")
                
                with col2:
                    st.write(f"**Enhanced Scores**:")
                    st.write(f"- Semantic Similarity: {result.semantic_similarity:.3f}")
                    st.write(f"- Keyword Coverage: {result.keyword_coverage:.3f}")
                    st.write(f"- Business Relevance: {getattr(result, 'business_relevance', 0.7):.3f}")
                    st.write(f"- Factual Accuracy: {getattr(result, 'factual_accuracy', 0.7):.3f}")
                    st.write(f"- Response Time: {result.response_time:.2f}s")

def enhanced_export_section():
    """Enhanced export section with fallback analytics."""
    
    st.header("📤 Enhanced Export & Analytics Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💬 Enhanced Chat History")
        if st.button("📥 Download Enhanced Chat History"):
            if st.session_state.chat_history:
                chat_data = []
                for message in st.session_state.chat_history:
                    chat_data.append({
                        'Timestamp': message['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'Role': message['role'],
                        'Content': message['content'],
                        'Response_Time': message.get('response_time', ''),
                        'Accuracy': message.get('accuracy', ''),
                        'Relevance': message.get('relevance', ''),
                        'Completeness': message.get('completeness', ''),
                        'Used_Fallback': message.get('used_fallback', False),
                        'Fallback_Strategy': message.get('fallback_strategy', ''),
                        'Fallback_Reason': message.get('fallback_reason', '')
                    })
                
                chat_df = pd.DataFrame(chat_data)
                csv = chat_df.to_csv(index=False)
                
                st.download_button(
                    label="📄 Download Enhanced CSV",
                    data=csv,
                    file_name=f"insightforge_enhanced_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No chat history available.")
    
    with col2:
        st.subheader("📊 Fallback Analytics")
        if st.button("📈 Generate Fallback Report"):
            if st.session_state.fallback_stats["total_queries"] > 0:
                fallback_data = {
                    'fallback_summary': {
                        'total_queries': st.session_state.fallback_stats["total_queries"],
                        'fallback_used': st.session_state.fallback_stats["fallback_used"],
                        'fallback_rate': (st.session_state.fallback_stats["fallback_used"] / 
                                        st.session_state.fallback_stats["total_queries"] * 100),
                        'average_response_time': np.mean([r.response_time for r in st.session_state.evaluation_results]) if st.session_state.evaluation_results else 0
                    },
                    'performance_by_source': {
                        'internal_only_queries': st.session_state.fallback_stats["total_queries"] - st.session_state.fallback_stats["fallback_used"],
                        'enhanced_with_fallback': st.session_state.fallback_stats["fallback_used"]
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                report_json = json.dumps(fallback_data, indent=2)
                
                st.download_button(
                    label="📊 Download Fallback Analytics",
                    data=report_json,
                    file_name=f"insightforge_fallback_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No fallback data available.")
    
    with col3:
        st.subheader("🔬 Complete Performance Report")
        if st.button("📋 Generate Complete Report"):
            if st.session_state.evaluation_results:
                complete_report = {
                    'system_summary': {
                        'total_queries': len(st.session_state.chat_history) // 2,
                        'avg_response_time': np.mean([r.response_time for r in st.session_state.evaluation_results]),
                        'avg_accuracy': np.mean([r.accuracy_score for r in st.session_state.evaluation_results]),
                        'avg_relevance': np.mean([r.relevance_score for r in st.session_state.evaluation_results]),
                        'fallback_usage_rate': (st.session_state.fallback_stats["fallback_used"] / 
                                              st.session_state.fallback_stats["total_queries"] * 100) if st.session_state.fallback_stats["total_queries"] > 0 else 0
                    },
                    'enhanced_evaluation_results': [
                        {
                            'question': result.question,
                            'accuracy_score': result.accuracy_score,
                            'relevance_score': result.relevance_score,
                            'completeness_score': result.completeness_score,
                            'semantic_similarity': result.semantic_similarity,
                            'keyword_coverage': result.keyword_coverage,
                            'business_relevance': getattr(result, 'business_relevance', 0.7),
                            'factual_accuracy': getattr(result, 'factual_accuracy', 0.7),
                            'response_time': result.response_time,
                            'evaluation_notes': result.evaluation_notes
                        }
                        for result in st.session_state.evaluation_results
                    ],
                    'fallback_statistics': st.session_state.fallback_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
                report_json = json.dumps(complete_report, indent=2)
                
                st.download_button(
                    label="📋 Download Complete Enhanced Report",
                    data=report_json,
                    file_name=f"insightforge_complete_enhanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No performance data available.")

def analytics_dashboard():
    """Display analytics dashboard (kept for compatibility)."""
    
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.sales_data is None:
        st.warning("No sales data available for visualization.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = st.session_state.sales_data['Sales'].sum()
    avg_sale = st.session_state.sales_data['Sales'].mean()
    total_records = len(st.session_state.sales_data)
    
    try:
        if 'Date' in st.session_state.sales_data.columns:
            dates = pd.to_datetime(st.session_state.sales_data['Date'])
            date_range = f"{dates.min().date()} to {dates.max().date()}"
        else:
            date_range = "Date range not available"
    except Exception:
        date_range = "Date range calculation error"
    
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Average Sale", f"${avg_sale:.0f}")
    col3.metric("Total Records", f"{total_records:,}")
    col4.metric("Date Range", date_range)
    
    # Simple visualization
    st.subheader("Sales by Product")
    if 'Product' in st.session_state.sales_data.columns:
        product_sales = st.session_state.sales_data.groupby('Product')['Sales'].sum()
        st.bar_chart(product_sales)

def main():
    """Enhanced main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Display enhanced header
    main_header()
    
    # Enhanced sidebar configuration
    enhanced_sidebar_configuration()
    
    # Initialize enhanced systems if not already done
    if not st.session_state.enhanced_initialized:
        with st.spinner("🚀 Initializing Enhanced InsightForge AI systems..."):
            # Create a basic enhanced pipeline for initialization
            basic_pipeline = EnhancedInsightForgeRAGPipeline(
                knowledge_base_path=os.path.join(project_root, "data", "knowledge_base"),
                llm=None,  # Will use mock LLM initially
                enable_fallback=st.session_state.fallback_enabled
            )
            
            systems = load_enhanced_insightforge_systems(basic_pipeline)
            if systems:
                st.session_state.enhanced_rag_pipeline = systems['enhanced_rag_pipeline']
                st.session_state.prompt_engine = systems['prompt_engine']
                st.session_state.memory_system = systems['memory_system']
                st.session_state.improved_evaluator = systems['improved_evaluator']
                st.session_state.visualizer = systems['visualizer']
                st.session_state.enhanced_initialized = True
                
                # Start memory session
                if st.session_state.memory_system:
                    session_id = st.session_state.memory_system.start_new_session("enhanced_streamlit_session")
                    st.session_state.current_session_id = session_id
                
                st.success("✅ Enhanced InsightForge AI systems initialized!")
            else:
                st.error("❌ Failed to initialize enhanced systems. Please check your configuration.")
                st.stop()
    
    # Load sales data if not already loaded
    if st.session_state.sales_data is None:
        st.session_state.sales_data = load_sales_data()
    
    # Enhanced application tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Enhanced AI Assistant", 
        "📊 Analytics", 
        "🧪 Enhanced Performance", 
        "📤 Enhanced Export"
    ])
    
    with tab1:
        enhanced_chat_interface()
    
    with tab2:
        analytics_dashboard()
    
    with tab3:
        enhanced_system_performance_tab()

    with tab4:
        enhanced_export_section()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚀 <strong>InsightForge AI Enhanced</strong> - Advanced Business Intelligence with External Knowledge Access</p>
        <p>Powered by RAG, Memory-Enhanced AI, Tavily Web Search, Wikipedia Integration & Advanced Evaluation</p>
        <p>🌐 <em>Now with intelligent fallback to external sources for comprehensive insights</em></p>
        <p>Built with ❤️ using Streamlit, LangChain, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()