# ui/streamlit_app.py - Complete Updated InsightForge AI Interface with Improved Evaluation

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
    from app.rag_pipeline import InsightForgeRAGPipeline
    from app.prompt_engineering import InsightForgePromptEngineering
    from app.memory import InsightForgeMemorySystem, ConversationContext
    from evaluation.evaluator import InsightForgeEvaluator
    from evaluation.improved_evaluator import create_improved_evaluator
    from utils.visualizations import InsightForgeVisualizer
    from loaders.data_loader import load_and_process_sales_data
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all InsightForge AI components are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
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
    .success-badge {
        background: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .warning-badge {
        background: #FF9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .error-badge {
        background: #F44336;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .source-badge {
        background: #6c757d;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 8px;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
    .metric-good {
        color: #4CAF50;
        font-weight: bold;
    }
    .metric-fair {
        color: #FF9800;
        font-weight: bold;
    }
    .metric-poor {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.rag_pipeline = None
        st.session_state.prompt_engine = None
        st.session_state.memory_system = None
        st.session_state.evaluator = None
        st.session_state.improved_evaluator = None
        st.session_state.visualizer = None
        st.session_state.chat_history = []
        st.session_state.evaluation_results = []
        st.session_state.sales_data = None
        st.session_state.system_performance = {}
        st.session_state.current_session_id = None
        st.session_state.llm_configured = False
        st.session_state.api_key_input = ""

def check_environment_variables():
    """Check for API keys in environment variables."""
    
    env_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    }
    
    return env_keys

def configure_llm_provider(provider: str, api_key: str = None):
    """Configure the selected LLM provider."""
    
    try:
        if provider == "OpenAI GPT":
            if not api_key:
                st.error("OpenAI API key is required")
                return None
            
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            return llm
            
        elif provider == "Anthropic Claude":
            if not api_key:
                st.error("Anthropic API key is required")
                return None
            
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    api_key=api_key,
                    model="claude-3-sonnet-20240229",
                    temperature=0.1
                )
                return llm
            except ImportError:
                st.error("Anthropic package not installed. Run: pip install langchain-anthropic")
                return None
        
        elif provider == "Local Model":
            try:
                from langchain_community.llms import Ollama
                llm = Ollama(
                    model="llama2",
                    temperature=0.1
                )
                return llm
            except ImportError:
                st.error("Ollama package not available. Please install or use other providers.")
                return None
        
        else:  # Mock LLM
            return None
            
    except Exception as e:
        st.error(f"Error configuring {provider}: {e}")
        return None

@st.cache_resource
def load_insightforge_systems(_llm=None):
    """Load and cache InsightForge AI systems with specified LLM."""
    
    try:
        kb_path = os.path.join(project_root, "data", "knowledge_base")
        
        systems = {
            'rag_pipeline': InsightForgeRAGPipeline(
                knowledge_base_path=kb_path,
                llm=_llm
            ),
            'prompt_engine': InsightForgePromptEngineering(),
            'memory_system': InsightForgeMemorySystem(
                memory_type="buffer_window",
                window_size=10,
                persist_memory=True
            ),
            'evaluator': InsightForgeEvaluator(),
            'improved_evaluator': create_improved_evaluator(),
            'visualizer': InsightForgeVisualizer()
        }
        
        return systems
        
    except Exception as e:
        st.error(f"Error loading InsightForge systems: {e}")
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
    """Display main application header."""
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 InsightForge AI</h1>
        <h3>Advanced Business Intelligence Platform</h3>
        <p>Ask any business question - I'll find insights from your data and strategic knowledge</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_configuration():
    """Configure sidebar with system settings and LLM selection."""
    
    st.sidebar.markdown("## ⚙️ System Configuration")
    
    # System status
    with st.sidebar.expander("🔍 System Status", expanded=True):
        if st.session_state.initialized and st.session_state.llm_configured:
            st.markdown('<span class="success-badge">✅ All Systems Operational</span>', unsafe_allow_html=True)
            st.write("🤖 RAG Pipeline: Ready")
            st.write("🧠 Memory System: Active") 
            st.write("📚 Unified Knowledge Base: Active")
            st.write("🎯 Improved Evaluation: Active")
        elif st.session_state.initialized:
            st.markdown('<span class="warning-badge">⚠️ LLM Configuration Needed</span>', unsafe_allow_html=True)
            st.write("🤖 RAG Pipeline: Ready (Mock Mode)")
        else:
            st.markdown('<span class="error-badge">🔄 Initializing Systems</span>', unsafe_allow_html=True)
    
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
                st.sidebar.success("🔑 API Key configured")
    
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
                st.sidebar.success("🔑 API Key configured")
    
    elif llm_option == "Local Model":
        st.sidebar.info("🖥️ Using local Ollama model")
        api_key = "local"
    
    else:  # Mock LLM
        st.sidebar.info("🧪 Using Mock LLM for testing")
        api_key = "mock"
    
    # Configure LLM button
    if st.sidebar.button("🚀 Configure LLM", type="primary"):
        if llm_option in ["Mock LLM (Testing)", "Local Model"] or api_key:
            with st.spinner("Configuring LLM..."):
                llm = configure_llm_provider(llm_option, api_key)
                
                # Clear cache and reload with new LLM
                st.cache_resource.clear()
                
                # Force reload of systems with the new LLM
                systems = load_insightforge_systems(llm)
                
                if systems:
                    st.session_state.rag_pipeline = systems['rag_pipeline']
                    st.session_state.prompt_engine = systems['prompt_engine']
                    st.session_state.memory_system = systems['memory_system']
                    st.session_state.evaluator = systems['evaluator']
                    st.session_state.improved_evaluator = systems['improved_evaluator']
                    st.session_state.visualizer = systems['visualizer']
                    st.session_state.llm_configured = True
                    
                    st.sidebar.success(f"✅ {llm_option} configured successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to configure LLM")
        else:
            st.sidebar.error("Please provide an API key")
    
    # Memory Settings
    st.sidebar.markdown("## 🧠 Memory Settings")
    
    if st.sidebar.button("🧹 Clear Memory"):
        if st.session_state.memory_system:
            st.session_state.memory_system.clear_session_memory()
            st.session_state.chat_history = []
            st.sidebar.success("Memory cleared!")
    
    # Knowledge Base Info
    st.sidebar.markdown("## 📚 Knowledge Base")
    st.sidebar.info("""
    **Unified Knowledge Sources:**
    • Sales data & analytics
    • Customer demographics
    • Product performance
    • Regional analysis
    • AI business strategies
    • Strategic recommendations
    """)

def format_metric_score(score: float, metric_name: str) -> str:
    """Format metric score with color coding."""
    
    if score >= 0.8:
        css_class = "metric-good"
        icon = "🟢"
    elif score >= 0.6:
        css_class = "metric-fair"
        icon = "🟡"
    else:
        css_class = "metric-poor"
        icon = "🔴"
    
    return f'{icon} <span class="{css_class}">{metric_name}: {score:.3f}</span>'



def chat_interface():
    """Main unified chat interface for all business intelligence queries."""
    
    st.header("💬 Ask Me Anything About Your Business")
    
    # Display configuration warning if needed
    if not st.session_state.llm_configured:
        st.warning("⚠️ Please configure an LLM provider in the sidebar to get AI-powered responses.")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history - CLEAN VERSION (NO HTML)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                # User message - simple clean display
                st.write("**You:**", message["content"])
                
            else:
                # Assistant message - clean display
                st.write("**InsightForge AI:**", message["content"])
                
                # Show sources in a clean expandable section
                if message.get("sources"):
                    with st.expander("📚 Sources Used"):
                        source_types = set()
                        for src in message["sources"]:
                            if "Sales Data" in src.get("source_type", ""):
                                source_types.add("📊 Sales Data")
                            elif "PDF" in src.get("source_type", ""):
                                source_types.add("📄 Business Documents")
                            else:
                                source_types.add("📋 Knowledge Base")
                        
                        for source_type in source_types:
                            st.write(f"• {source_type}")
                
                # Show metrics using Streamlit's native metric display
                if message.get("response_time"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Response Time", f"{message.get('response_time', 0):.2f}s")
                    
                    with col2:
                        accuracy = message.get("accuracy", 0)
                        color = "🟢" if accuracy >= 0.8 else "🟡" if accuracy >= 0.6 else "🔴"
                        st.metric("Accuracy", f"{accuracy:.3f}", delta=color)
                    
                    with col3:
                        relevance = message.get("relevance", 0)
                        color = "🟢" if relevance >= 0.8 else "🟡" if relevance >= 0.6 else "🔴"
                        st.metric("Relevance", f"{relevance:.3f}", delta=color)
                    
                    with col4:
                        completeness = message.get("completeness", 0)
                        color = "🟢" if completeness >= 0.8 else "🟡" if completeness >= 0.6 else "🔴"
                        st.metric("Completeness", f"{completeness:.3f}", delta=color)
            
            # Clean separator
            st.write("---")
    
    # Enhanced Quick Questions
    st.markdown("### 🚀 Try These Questions")
    st.markdown("*The system automatically finds relevant information from all sources*")
    
    # Create a natural grid of questions
    col1, col2, col3 = st.columns(3)
    
    # Mix different types of questions naturally
    unified_questions = [
        "What are our total sales figures?",
        "Which products are performing best?", 
        "How do different regions compare?",
        "What customer trends do you see?",
        "What strategic recommendations do you have?",
        "How can we improve our performance?"
    ]
    
    # Display questions in columns
    for i, question in enumerate(unified_questions):
        col = [col1, col2, col3][i % 3]
        if col.button(question, key=f"q_{i}", use_container_width=True):
            process_user_query(question)
    
    # Chat input
    user_query = st.chat_input("Ask me anything about your business data...")
    
    if user_query:
        process_user_query(user_query)



def process_user_query(query: str):
    """Process user query through the InsightForge AI pipeline with improved evaluation."""
    
    if not st.session_state.initialized:
        st.error("Please wait for system initialization to complete.")
        return
    
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now()
    })
    
    # Show processing indicator
    with st.spinner("🤖 InsightForge AI is analyzing your question..."):
        
        try:
            start_time = time.time()
            
            # Step 1: Get response from RAG pipeline
            response = st.session_state.rag_pipeline.query(query, return_sources=True)
            
            # Step 2: Enhance with memory context if available
            if st.session_state.memory_system:
                memory_context = st.session_state.memory_system.get_conversation_context()
                if memory_context.get('recent_topics'):
                    response['answer'] += f"\n\n*Building on our previous discussion about {', '.join(memory_context['recent_topics'][-2:])}, this analysis provides additional insights.*"
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Step 3: Evaluate response quality with IMPROVED evaluator
            eval_result = None
            try:
                if st.session_state.improved_evaluator:
                    eval_result = st.session_state.improved_evaluator.evaluate_response(
                        question=query,
                        predicted_answer=response.get('answer', ''),
                        reference_answer=None,  # No reference needed for improved evaluator
                        response_time=response_time,
                        sources_used=[src.get('source_type', '') for src in response.get('sources', [])]
                    )
                    
                    # Store the improved result
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
            
            # Add assistant response to chat with improved metrics
            assistant_message = {
                "role": "assistant",
                "content": response.get('answer', 'I apologize, but I could not generate a response.'),
                "timestamp": datetime.now(),
                "response_time": response_time,
                "accuracy": eval_result.accuracy_score if eval_result else 0,
                "relevance": eval_result.relevance_score if eval_result else 0,
                "completeness": eval_result.completeness_score if eval_result else 0,
                "sources": response.get('sources', [])
            }
            
            st.session_state.chat_history.append(assistant_message)
            
            # Update system performance metrics
            update_performance_metrics(response_time, eval_result)
            
        except Exception as e:
            error_message = f"I encountered an error while processing your question: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_message,
                "timestamp": datetime.now()
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
    
    # Look for product/region names
    if 'Product' in response:
        insights.append("Product performance analysis")
    if any(word in response for word in ['North', 'South', 'East', 'West']):
        insights.append("Regional analysis insights")
    
    return insights[:3]

def update_performance_metrics(response_time: float, eval_result: Any):
    """Update system performance metrics with improved evaluation results."""
    
    if 'response_times' not in st.session_state.system_performance:
        st.session_state.system_performance['response_times'] = []
        st.session_state.system_performance['accuracy_scores'] = []
        st.session_state.system_performance['relevance_scores'] = []
        st.session_state.system_performance['completeness_scores'] = []
        st.session_state.system_performance['total_queries'] = 0
    
    st.session_state.system_performance['response_times'].append(response_time)
    st.session_state.system_performance['total_queries'] += 1
    
    if eval_result:
        st.session_state.system_performance['accuracy_scores'].append(eval_result.accuracy_score)
        st.session_state.system_performance['relevance_scores'].append(eval_result.relevance_score)
        st.session_state.system_performance['completeness_scores'].append(eval_result.completeness_score)

def analytics_dashboard():
    """Display analytics and performance dashboard."""
    
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.sales_data is None:
        st.warning("No sales data available for visualization.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = st.session_state.sales_data['Sales'].sum()
    avg_sale = st.session_state.sales_data['Sales'].mean()
    total_records = len(st.session_state.sales_data)
    
    # Safe date range calculation
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

def system_performance_tab():
    """Display improved system performance metrics."""
    
    st.header("🧪 System Performance - Improved Evaluation")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation data available yet. Start asking questions to see performance metrics!")
        return
    
    # Performance overview with improved metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
        
        # Detailed breakdown
        st.subheader("📈 Detailed Metrics")
        
        # Create a dataframe for better visualization
        metrics_data = []
        for i, result in enumerate(results[-10:], 1):  # Last 10 results
            metrics_data.append({
                'Query': f"Q{i}",
                'Accuracy': result.accuracy_score,
                'Relevance': result.relevance_score,
                'Completeness': result.completeness_score,
                'Business_Relevance': result.business_relevance,
                'Factual_Accuracy': result.factual_accuracy
            })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            st.line_chart(df.set_index('Query'))
            
            # Show evaluation notes for recent queries
            st.subheader("📝 Recent Evaluation Notes")
            for i, result in enumerate(results[-3:], 1):  # Last 3 results
                with st.expander(f"Query {len(results)-3+i}: {result.question[:50]}..."):
                    st.write(f"**Answer**: {result.predicted_answer[:200]}...")
                    st.write(f"**Evaluation Notes**: {result.evaluation_notes}")
                    st.write(f"**Detailed Scores**:")
                    st.write(f"- Semantic Similarity: {result.semantic_similarity:.3f}")
                    st.write(f"- Keyword Coverage: {result.keyword_coverage:.3f}")
                    st.write(f"- Business Relevance: {result.business_relevance:.3f}")
                    st.write(f"- Factual Accuracy: {result.factual_accuracy:.3f}")

def export_section():
    """Export and reporting section with improved evaluation data."""
    
    st.header("📤 Export & Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💬 Chat History")
        if st.button("📥 Download Chat History"):
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
                        'Completeness': message.get('completeness', '')
                    })
                
                chat_df = pd.DataFrame(chat_data)
                csv = chat_df.to_csv(index=False)
                
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"insightforge_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No chat history available.")
    
    with col2:
        st.subheader("📊 Performance Report")
        if st.button("📈 Generate Performance Report"):
            if st.session_state.evaluation_results:
                
                report_data = {
                    'session_summary': {
                        'total_queries': len(st.session_state.chat_history) // 2,
                        'avg_response_time': np.mean([r.response_time for r in st.session_state.evaluation_results]),
                        'avg_accuracy': np.mean([r.accuracy_score for r in st.session_state.evaluation_results]),
                        'avg_relevance': np.mean([r.relevance_score for r in st.session_state.evaluation_results]),
                        'avg_completeness': np.mean([r.completeness_score for r in st.session_state.evaluation_results]),
                        'timestamp': datetime.now().isoformat()
                    },
                    'detailed_evaluation_results': [
                        {
                            'question': result.question,
                            'accuracy_score': result.accuracy_score,
                            'relevance_score': result.relevance_score,
                            'completeness_score': result.completeness_score,
                            'semantic_similarity': result.semantic_similarity,
                            'keyword_coverage': result.keyword_coverage,
                            'business_relevance': result.business_relevance,
                            'factual_accuracy': result.factual_accuracy,
                            'response_time': result.response_time,
                            'evaluation_notes': result.evaluation_notes
                        }
                        for result in st.session_state.evaluation_results
                    ]
                }
                
                report_json = json.dumps(report_data, indent=2)
                
                st.download_button(
                    label="📊 Download JSON Report",
                    data=report_json,
                    file_name=f"insightforge_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No performance data available.")

def debug_knowledge_base():
    """Debug function to see what's in the knowledge base."""
    
    st.header("🔍 Knowledge Base Debug")
    
    if st.button("🧪 Debug Knowledge Base Contents"):
        if st.session_state.rag_pipeline and st.session_state.rag_pipeline.vector_store:
            # Get some sample documents
            
            # Try different searches to find sales summary
            test_docs1 = st.session_state.rag_pipeline.vector_store.similarity_search("sales_summary", k=5)
            test_docs2 = st.session_state.rag_pipeline.vector_store.similarity_search("total sales", k=5)
            test_docs3 = st.session_state.rag_pipeline.vector_store.similarity_search("revenue", k=5)

            # Combine and deduplicate
            all_test_docs = test_docs1 + test_docs2 + test_docs3
            test_docs = list({doc.page_content: doc for doc in all_test_docs}.values())[:10]
                        
            
            st.subheader("📊 Sample Documents in Knowledge Base:")
            
            sales_count = 0
            pdf_count = 0
            
            for i, doc in enumerate(test_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                
                # Show detailed metadata for debugging
                st.write(f"**Doc {i+1}**:")
                st.write(f"**Source**: {metadata.get('source')}")
                st.write(f"**Source Type**: {metadata.get('source_type')}")
                st.write(f"**Section**: {metadata.get('section')}")
                st.write(f"**All Metadata**: {metadata}")
                
                # Count source types
                if (metadata.get('source_type') == 'sales_data' or 
                    metadata.get('section') or 
                    'sales_summary' in metadata.get('source', '')):
                    sales_count += 1
                    source_type = "📊 SALES DATA"
                else:
                    pdf_count += 1
                    source_type = "📄 PDF DATA"
                
                st.write(f"**Classified as**: {source_type}")
                st.write(f"**Content**: {content_preview}")
                st.write("---")
            
            st.write(f"**Summary**: {sales_count} sales docs, {pdf_count} PDF docs")
        else:
            st.error("No vector store available")

def test_improved_evaluation_display():
    """Display test results for the improved evaluation."""
    
    st.header("🧪 Test Improved Evaluation")
    
    if st.button("🎯 Run Evaluation Test"):
        try:
            # Test with a typical business intelligence response
            test_question = "What are our total sales figures?"
            test_answer = "Based on the sales data analysis, our total sales are $125,000 with strong performance in Product C ($50,000) and the West region ($55,000). The average sale is $85 with 65% female customers in the 26-35 age group showing the highest engagement."
            
            evaluator = create_improved_evaluator()
            result = evaluator.evaluate_response(
                question=test_question,
                predicted_answer=test_answer,
                reference_answer=None,  # No reference needed
                response_time=1.2
            )
            
            st.success("✅ Evaluation Test Completed!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{result.accuracy_score:.3f}")
            col2.metric("Relevance", f"{result.relevance_score:.3f}")
            col3.metric("Completeness", f"{result.completeness_score:.3f}")
            
            st.subheader("📝 Detailed Results")
            st.write(f"**Question**: {result.question}")
            st.write(f"**Answer**: {result.predicted_answer[:200]}...")
            st.write(f"**Evaluation Notes**: {result.evaluation_notes}")
            
            st.subheader("🔍 Component Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **Semantic Similarity**: {result.semantic_similarity:.3f}")
                st.write(f"- **Keyword Coverage**: {result.keyword_coverage:.3f}")
            with col2:
                st.write(f"- **Business Relevance**: {result.business_relevance:.3f}")
                st.write(f"- **Factual Accuracy**: {result.factual_accuracy:.3f}")
            
        except Exception as e:
            st.error(f"Evaluation test failed: {e}")

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    main_header()
    
    # Sidebar configuration
    sidebar_configuration()
    
    # Initialize systems if not already done
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing InsightForge AI systems..."):
            systems = load_insightforge_systems()
            if systems:
                st.session_state.rag_pipeline = systems['rag_pipeline']
                st.session_state.prompt_engine = systems['prompt_engine']
                st.session_state.memory_system = systems['memory_system']
                st.session_state.evaluator = systems['evaluator']
                st.session_state.improved_evaluator = systems['improved_evaluator']
                st.session_state.visualizer = systems['visualizer']
                st.session_state.initialized = True
                
                # Start memory session
                session_id = st.session_state.memory_system.start_new_session("streamlit_session")
                st.session_state.current_session_id = session_id
                
                st.success("✅ InsightForge AI systems initialized successfully!")
            else:
                st.error("❌ Failed to initialize systems. Please check your configuration.")
                st.stop()
    
    # Load sales data if not already loaded
    if st.session_state.sales_data is None:
        st.session_state.sales_data = load_sales_data()
    
    # Main application tabs (Debug tab hidden but code preserved)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Assistant", "📊 Analytics", "🧪 Performance", "📤 Export", "🎯 Test Evaluation"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        analytics_dashboard()
    
    with tab3:
        system_performance_tab()

    with tab4:
        export_section()

    # Debug tab hidden but function preserved for future use
    # with tab5:
    #     debug_knowledge_base()

    with tab5:  # Test Evaluation moved to tab5
        test_improved_evaluation_display()

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚀 <strong>InsightForge AI</strong> - Advanced Business Intelligence Platform</p>
        <p>Powered by RAG, Memory-Enhanced AI, Advanced Analytics & Improved Evaluation System</p>
        <p>Built with ❤️ using Streamlit, LangChain, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()