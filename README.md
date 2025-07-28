# InsightForge AI

Advanced Business Intelligence Platform powered by RAG, Memory-Enhanced AI, and Interactive Analytics.

## Features

- AI Chat Assistant - Ask natural language questions about your business data
- Interactive Analytics - Dynamic visualizations for sales, products, regions, customers
- Memory Integration - AI remembers conversation context for better insights
- Performance Monitoring - Track AI response quality and accuracy
- Export Capabilities - Download reports, charts, and conversation history

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Knowledge Base:**
   ```bash
   cd loaders
   python simple_pdf_test.py
   ```

3. **Launch Application:**
   ```bash
   python main.py
   ```

## Project Structure

```
insightforge_ai/
├── app/                    # Core AI components
├── loaders/               # Data loading and processing
├── evaluation/            # Model evaluation and QA
├── ui/                    # Streamlit user interface
├── utils/                 # Utilities and visualizations
├── data/                  # Data files and outputs
└── tests/                 # Testing scripts
```

## Configuration

- Copy `.env.template` to `.env` and add your API keys
- Configure LLM providers in the Streamlit sidebar
- Customize visualization themes and settings

## Sample Queries

- "What are our total sales figures?"
- "Which products are performing best?"
- "How do different regions compare?"
- "What customer trends do you see?"
- "Based on our data, what strategic recommendations do you have?"

## Next Steps

1. Upload your business data (CSV format)
2. Add relevant PDF documents to `data/pdfs/`
3. Configure your preferred LLM provider
4. Start exploring your data with AI-powered insights!

---

Built with LangChain, Streamlit, and Plotly
