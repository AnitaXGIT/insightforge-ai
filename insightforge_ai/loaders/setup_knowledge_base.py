# loaders/setup_knowledge_base.py - InsightForge AI Knowledge Base Setup

import os
import sys
from pathlib import Path
import glob

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Debug: Print paths to see what's happening
print(f"Current dir: {current_dir}")
print(f"Project root: {project_root}")
print(f"Looking for data at: {project_root / 'data'}")


from pdf_loader import SalesKnowledgeBaseCreator
from data_loader import load_and_process_sales_data

def check_existing_knowledge_base():
    """Check if knowledge base already exists."""
    
    kb_path = project_root / "data" / "knowledge_base"
    faiss_path = kb_path / "faiss_index"
    
    if faiss_path.exists() and any(faiss_path.iterdir()):
        return True, kb_path
    else:
        return False, kb_path

def scan_pdf_files():
    """Scan for PDF files in the data/pdfs directory."""
    
    pdf_dir = project_root / "data" / "pdfs"
    
    if not pdf_dir.exists():
        print(f"📁 Creating PDF directory: {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        file_size = pdf_file.stat().st_size / 1024  # KB
        print(f"   • {pdf_file.name} ({file_size:.1f} KB)")
    
    return pdf_files

def check_sales_data():
    """Check if sales data exists."""
    
    sales_path = project_root / "data" / "sales_data.csv"
    
    if sales_path.exists():
        import pandas as pd
        df = pd.read_csv(sales_path)
        print(f"📊 Sales data found: {len(df)} records")
        return True, sales_path
    else:
        print("📊 No sales data found, will create sample data")
        return False, sales_path

def create_sample_ai_document():
    """Create a sample AI business document if no PDFs exist."""
    
    ai_content = """AI BUSINESS MODEL INNOVATIONS FOR ENTERPRISE SUCCESS

EXECUTIVE SUMMARY
This document outlines proven AI business model innovations that drive measurable improvements in sales performance, customer satisfaction, and operational efficiency.

SALES PERFORMANCE OPTIMIZATION
AI-Driven Sales Enhancement:
• Predictive lead scoring improves conversion rates by 35-50%
• Dynamic pricing optimization increases revenue by 15-25%
• Sales forecasting accuracy improves by 40% with ML models
• Automated pipeline management reduces sales cycle time by 25%

Revenue Growth Strategies:
• Cross-selling AI algorithms boost average order value by 20%
• Customer lifetime value prediction enables targeted retention
• Territory optimization using geographic AI analysis
• Competitive pricing intelligence through market analysis

CUSTOMER INTELLIGENCE TRANSFORMATION
Advanced Customer Analytics:
• Real-time behavioral segmentation for personalized experiences
• Sentiment analysis drives 30% improvement in satisfaction scores
• Churn prediction models achieve 85% accuracy in early detection
• Recommendation engines increase engagement by 40%

Customer Experience Innovation:
• AI chatbots handle 70% of routine customer inquiries
• Personalization engines deliver individualized customer journeys
• Predictive service delivery anticipates customer needs
• Voice of customer analysis guides product development

PRODUCT PERFORMANCE ENHANCEMENT
AI-Powered Product Strategy:
• Market demand prediction with 90% accuracy using trend analysis
• Quality control automation reduces defects by 60%
• Intelligent inventory management prevents stockouts
• Product recommendation systems increase cross-selling success

Innovation Management:
• Customer feedback analysis prioritizes feature development
• Competitive intelligence guides positioning strategies
• Price elasticity modeling optimizes product pricing
• Performance analytics identify improvement opportunities

REGIONAL EXPANSION INTELLIGENCE
Geographic Growth Strategy:
• Market sizing analysis identifies high-potential regions
• Cultural adaptation AI guides localization strategies
• Regional preference modeling personalizes offerings
• Expansion risk assessment prevents costly mistakes

Location-Based Optimization:
• Site selection algorithms for retail expansion
• Distribution network optimization reduces costs by 20%
• Regional pricing strategies maximize market penetration
• Local competition analysis informs market entry

STRATEGIC IMPLEMENTATION FRAMEWORK
Phase 1: Foundation Building (Months 1-3)
• Establish data governance and quality standards
• Implement unified data platform architecture
• Deploy basic analytics and reporting infrastructure
• Create AI ethics and governance framework

Phase 2: Core AI Integration (Months 4-9)
• Implement customer segmentation and scoring models
• Deploy sales forecasting and pipeline optimization
• Launch recommendation engines and personalization
• Create predictive maintenance and quality systems

Phase 3: Advanced Optimization (Months 10-18)
• Implement autonomous decision-making systems
• Deploy advanced predictive analytics across functions
• Create AI-driven strategic planning capabilities
• Launch innovation and product development AI

MEASURABLE BUSINESS IMPACT
Revenue Metrics:
• 25-40% increase in qualified lead conversion rates
• 15-30% improvement in customer retention rates
• 20-35% growth in average revenue per customer
• 10-25% reduction in customer acquisition costs

Operational Efficiency:
• 50-70% reduction in manual reporting time
• 30-50% improvement in forecast accuracy
• 40-60% decrease in operational costs
• 60-80% faster decision-making cycles

Customer Satisfaction:
• 30-45% improvement in customer satisfaction scores
• 40-60% reduction in service response times
• 25-40% increase in customer engagement metrics
• 35-50% growth in customer lifetime value

TECHNOLOGY REQUIREMENTS
Core AI Platform Components:
• Machine learning model management and deployment
• Real-time data processing and analytics engine
• Automated model training and optimization
• Scalable cloud infrastructure with elastic compute

Integration Requirements:
• CRM and sales automation platform connectivity
• ERP and financial systems data synchronization
• Marketing automation and campaign management
• Customer service and support platform integration

RISK MITIGATION AND SUCCESS FACTORS
Critical Success Elements:
• Executive leadership commitment and sponsorship
• Cross-functional team collaboration and communication
• Comprehensive change management and training programs
• Continuous monitoring and optimization processes

Risk Mitigation Strategies:
• Robust data privacy and security frameworks
• Comprehensive testing and validation protocols
• Gradual rollout with pilot programs and feedback loops
• Regular performance monitoring and course correction

CONCLUSION AND NEXT STEPS
AI business model innovations represent a fundamental shift in how organizations create value, serve customers, and compete in the market. Success requires strategic vision, tactical execution, and continuous adaptation.

The organizations that embrace these AI-driven approaches will achieve sustainable competitive advantages through superior customer insights, operational efficiency, and strategic agility.

Key next steps include establishing data foundations, implementing core AI capabilities, and building organizational competencies for AI-driven transformation."""

    return ai_content

def setup_knowledge_base():
    """Main function to set up the knowledge base properly."""
    
    print("🚀 InsightForge AI - Knowledge Base Setup")
    print("=" * 60)
    
    # Step 1: Check if knowledge base already exists
    kb_exists, kb_path = check_existing_knowledge_base()
    
    if kb_exists:
        print("📚 Existing knowledge base found!")
        user_input = input("Do you want to rebuild it? (y/n): ").lower().strip()
        
        if user_input != 'y':
            print("✅ Using existing knowledge base")
            print(f"📍 Location: {kb_path}")
            print("\n🚀 Ready to launch! Run: python main.py")
            return True
        else:
            print("🔄 Rebuilding knowledge base...")
    
    # Step 2: Initialize knowledge base creator
    kb_creator = SalesKnowledgeBaseCreator()
    
    # Step 3: Scan for PDF files
    print("\n📄 Step 1: Scanning for PDF documents...")
    pdf_files = scan_pdf_files()
    
    if not pdf_files:
        print("⚠️  No PDF files found in data/pdfs/")
        create_sample = input("Create sample AI business document? (y/n): ").lower().strip()
        
        if create_sample == 'y':
            print("📝 Creating sample AI business document...")
            ai_content = create_sample_ai_document()
            
            pdf_dir = project_root / "data" / "pdfs"
            sample_file = pdf_dir / "AI_Business_Innovation_Sample.txt"
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(ai_content)
            
            print(f"✅ Sample document created: {sample_file.name}")
            pdf_files = [sample_file]  # Treat as PDF for processing
        else:
            print("📋 Continuing without PDF documents...")
            pdf_files = []
    
    # Step 4: Check sales data
    print("\n📊 Step 2: Checking sales data...")
    sales_exists, sales_path = check_sales_data()
    
    # Step 5: Build knowledge base
    print("\n🧠 Step 3: Building knowledge base...")
    
    try:
        # Determine what to include
        include_sales = sales_exists
        pdf_paths = [str(pdf) for pdf in pdf_files] if pdf_files else None
        
        if not include_sales and not pdf_paths:
            print("❌ No data sources found! Please add PDF files or sales data.")
            return False
        
        print(f"📋 Building knowledge base with:")
        if include_sales:
            print(f"   • Sales data summary from {sales_path.name}")
        if pdf_paths:
            print(f"   • {len(pdf_paths)} document(s) from data/pdfs/")
        
        # Create the knowledge base
        vector_store = kb_creator.create_complete_knowledge_base(
            summary_path=str(sales_path) if sales_exists else None,
            save_path=str(kb_path),
            pdf_paths=pdf_paths,
            pdf_directory=None,  # We already scanned manually
            include_sales_summary=include_sales
        )
        
        print(f"✅ Knowledge base created successfully!")
        print(f"📍 Saved to: {kb_path}")
        print(f"📈 Total vectors: {vector_store.index.ntotal}")
        
    except Exception as e:
        print(f"❌ Error building knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test the knowledge base
    print("\n🧪 Step 4: Testing knowledge base...")
    
    test_queries = [
        "What are our total sales figures?",
        "How can AI improve our business?",
        "What are the key business recommendations?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: {query}")
        try:
            results = kb_creator.test_retrieval(query, k=2)
            print(f"   ✅ Retrieved {len(results)} relevant documents")
            
            # Show source types
            sources = set()
            for doc in results:
                if doc.metadata.get('section'):
                    sources.add("Sales Data")
                elif doc.metadata.get('source_type') == 'pdf':
                    sources.add("Business Documents")
                else:
                    sources.add("Knowledge Base")
            
            print(f"   📊 Sources: {', '.join(sources)}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    # Step 7: Create usage instructions
    create_usage_instructions()
    
    print("\n🎉 KNOWLEDGE BASE SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Your knowledge base is ready and contains:")
    
    if include_sales:
        print("   📊 Sales data analytics and insights")
    if pdf_files:
        print(f"   📄 {len(pdf_files)} business document(s)")
    
    print("\n🎯 WHAT'S NEXT:")
    print("1. Launch the application: python main.py")
    print("2. Configure your LLM provider in the sidebar")
    print("3. Start asking business questions!")
    print("\n💡 The system will automatically find relevant information")
    print("   from all sources without you needing to specify!")
    
    return True

def create_usage_instructions():
    """Create usage instructions file."""
    
    instructions = """# InsightForge AI - Usage Instructions

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
"""
    
    instructions_path = project_root / "data" / "KNOWLEDGE_BASE_INSTRUCTIONS.md"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"📖 Instructions saved: {instructions_path.name}")

def main():
    """Main execution function."""
    
    try:
        success = setup_knowledge_base()
        
        if success:
            print("\n🎊 SUCCESS! Your knowledge base is ready to use.")
        else:
            print("\n❌ Setup failed. Please check the errors and try again.")
            
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()