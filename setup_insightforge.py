# setup_insightforge.py - Setup script for InsightForge AI

import os
import sys
import subprocess

def create_directory_structure():
    """Create the complete InsightForge AI directory structure within the project folder."""
    
    print("🏗️ Creating InsightForge AI directory structure...")
    
    # Ensure we're working in the correct project directory
    project_name = "insightforge_ai"
    
    # Check if we're already in the insightforge_ai directory
    current_dir = os.getcwd()
    if not current_dir.endswith(project_name):
        # If not, create or navigate to insightforge_ai directory
        if not os.path.exists(project_name):
            os.makedirs(project_name)
            print(f"📁 Created project directory: {project_name}")
        os.chdir(project_name)
        print(f"📂 Working in: {os.getcwd()}")
    
    directories = [
        'app',
        'loaders', 
        'evaluation',
        'ui',
        'utils',
        'tests',
        'data',
        'data/pdfs',
        'data/knowledge_base',
        'data/evaluation_results',
        'data/visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")
        
        # Create __init__.py files for Python packages
        if directory in ['app', 'loaders', 'evaluation', 'ui', 'utils']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'"""InsightForge AI - {directory} package"""\n')
                print(f"✅ {init_file}")



def install_dependencies():
    """Install required Python dependencies with safe version ranges."""
    
    print("\n📦 Installing Python dependencies with safe version ranges...")
    
    # Using version ranges that allow updates while preventing breaking changes
    requirements = [
        'streamlit>=1.25.0,<2.0.0',
        'pandas>=1.5.0,<3.0.0',
        'numpy>=1.24.0,<2.0.0', 
        'plotly>=5.15.0,<6.0.0',
        'langchain>=0.1.0,<1.0.0',
        'langchain-community>=0.0.20,<1.0.0',
        'sentence-transformers>=2.2.0,<3.0.0',
        'faiss-cpu>=1.7.4,<2.0.0',
        'matplotlib>=3.7.0,<4.0.0',
        'seaborn>=0.12.0,<1.0.0',
        'scikit-learn>=1.3.0,<2.0.0',
        'pypdf>=3.12.0,<4.0.0',
        'python-dotenv>=1.0.0,<2.0.0',
        'transformers>=4.30.0,<5.0.0',
        'torch>=2.0.0,<3.0.0'
    ]
    
    print(f"Installing {len(requirements)} packages with version constraints...")
    
    try:
        for package in requirements:
            print(f"📥 Installing {package.split('>=')[0]}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True, capture_output=True, text=True)
            
            # Check if package was already satisfied or newly installed
            if "already satisfied" in result.stdout.lower():
                print(f"✅ {package.split('>=')[0]} (already satisfied)")
            else:
                print(f"✅ {package.split('>=')[0]} (installed)")
        
        print("\n🎉 All dependencies installed successfully with safe version ranges!")
        print("💡 These version ranges allow security updates while preventing breaking changes")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        print("\n🔧 Try installing manually with:")
        print("pip install -r requirements.txt")
        return False

def create_sample_files():
    """Create sample configuration and data files."""
    
    print("\n📄 Creating sample configuration files...")
    
    # Create requirements.txt
    requirements_content = """# InsightForge AI Requirements
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
langchain>=0.1.0
langchain-community>=0.0.20
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pypdf>=3.12.0
python-dotenv>=1.0.0
transformers>=4.30.0
torch>=2.0.0

# Optional LLM providers
# langchain-openai>=0.0.5
# langchain-anthropic>=0.1.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("✅ requirements.txt")
    
    # Create .env template
    env_template = """# InsightForge AI Environment Variables
# Uncomment and add your API keys as needed

# OpenAI Configuration
# OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration  
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Other Configuration
# MODEL_TEMPERATURE=0.1
# MAX_TOKENS=2000
"""
    
    with open('.env.template', 'w', encoding='utf-8') as f:
        f.write(env_template)
    print("✅ .env.template")
    
    # Create README with safe encoding
    readme_content = """# InsightForge AI

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
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ README.md")

def show_next_steps():
    """Show next steps after setup."""
    
    print("\n🎉 INSIGHTFORGE AI SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Directory structure created")
    print("✅ Dependencies installed") 
    print("✅ Configuration files created")
    print()
    print("🎯 NEXT STEPS:")
    print("1. Add your business data:")
    print("   • Place CSV files in data/")
    print("   • Add PDF documents to data/pdfs/")
    print()
    print("2. Create the knowledge base:")
    print("   cd loaders")
    print("   python simple_pdf_test.py")
    print()
    print("3. (Optional) Configure API keys:")
    print("   • Copy .env.template to .env")
    print("   • Add OpenAI or Anthropic API keys")
    print()
    print("4. Launch InsightForge AI:")
    print("   python main.py")
    print()
    print("🚀 Ready to explore your business data with AI!")

def main():
    """Main setup function."""
    
    print("🚀 InsightForge AI - Setup Wizard")
    print("=" * 60)
    print("Setting up your Advanced Business Intelligence Platform...")
    print()
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Install dependencies
        deps_installed = install_dependencies()
        if not deps_installed:
            print("⚠️ Some dependencies may not have installed correctly")
            print("Please check the error messages above")
        
        # Create sample files
        create_sample_files()
        
        # Show next steps
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()