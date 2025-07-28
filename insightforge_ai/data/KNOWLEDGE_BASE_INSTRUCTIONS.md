# InsightForge AI - Usage Instructions

## 🎯 Your Knowledge Base Contains

### Automatically Loaded Sources:
- **Sales Data**: Performance metrics, trends, customer analytics
- **PDF Documents**: All PDF files from data/pdfs/ folder
- **Business Intelligence**: Strategic insights and recommendations

## 💬 How to Use

### Natural Language Queries
Just ask questions naturally! The system automatically searches all sources:

**Sales Questions:**
- "What are our total sales figures?"
- "Which products perform best?"
- "How do regions compare?"

**Strategic Questions:**
- "How can AI improve our business?"
- "What are the key recommendations?"
- "How should we optimize performance?"

**Combined Questions:**
- "Based on our sales data, what AI strategies would help?"
- "How can we improve customer satisfaction using AI?"

## 🔄 Updating Your Knowledge Base

### Adding New Documents:
1. Place new PDF or text files in `data/pdfs/` folder
2. Run: `python loaders/simple_pdf_test.py`
3. Choose to rebuild when prompted

### Updating Sales Data:
1. Update `data/sales_data.csv` with new data
2. Run: `python loaders/simple_pdf_test.py`
3. Rebuild to include new sales insights

## 📊 What Gets Processed

### PDF Documents:
- All content is extracted and embedded
- Metadata preserved for source tracking
- Chunked for optimal retrieval

### Text Files:
- Processed as business documents
- Content embedded in vector store
- Treated same as PDF content

### Sales Data:
- Automatically analyzed for trends
- Key metrics calculated
- Insights generated and embedded

## 🚀 Ready to Explore!

Your knowledge base is persistent and will load automatically when you start the application. No need to rebuild unless you add new documents.

Launch with: `python main.py`

**Sample Questions:**
- "What are our total sales figures?"
- "How can AI improve our business?"
- "Based on our sales data, what AI strategies would help?"
- "How can we improve customer satisfaction using AI?"

## 🔄 Updating Your Knowledge Base

### Adding New PDF Documents:
1. Place new PDF files in `data/pdfs/` folder
2. Run: `python loaders/simple_pdf_test.py`
3. Choose to rebuild when prompted

### Updating Sales Data:
1. Update `data/sales_data.csv` with new data
2. Run: `python loaders/simple_pdf_test.py`
3. Rebuild to include new sales insights

## 📊 What Gets Processed

### PDF Documents:
- All content is extracted and embedded
- Metadata preserved for source tracking
- Chunked for optimal retrieval

### Sales Data:
- Automatically analyzed for trends
- Key metrics calculated
- Insights generated and embedded

## 🚀 Ready to Explore!

Your knowledge base is persistent and will load automatically when you start the application. No need to rebuild unless you add new documents.

Launch with: `python main.py`
