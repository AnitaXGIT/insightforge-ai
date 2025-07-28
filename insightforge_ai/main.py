# main.py - Enhanced InsightForge AI Entry Point

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if enhanced dependencies are installed."""
    try:
        import tavily
        import wikipedia
        import langchainhub
        return True
    except ImportError:
        return False

def check_knowledge_base():
    """Check if knowledge base exists."""
    kb_path = Path("data/knowledge_base/faiss_index")
    return kb_path.exists()

def main():
    """Main entry point for InsightForge AI."""
    
    print("🚀 InsightForge AI - Advanced Business Intelligence Platform")
    print("=" * 60)
    
    # Check knowledge base
    if not check_knowledge_base():
        print("❌ Knowledge base not found!")
        print("Please run the knowledge base setup first:")
        print("  cd loaders")
        print("  python setup_knowledge_base.py")
        return
    
    # Check enhanced dependencies
    enhanced_available = check_dependencies()
    
    print("📋 Available Systems:")
    print("1. 📊 Original InsightForge AI (Internal knowledge only)")
    
    if enhanced_available:
        print("2. 🌟 Enhanced InsightForge AI (Internal + External fallback)")
        print("3. 🎛️ System Configuration")
    else:
        print("2. 🔧 Install Enhanced Features")
    
    print("4. 🚪 Exit")
    print()
    
    try:
        choice = input("Choose an option (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching Original InsightForge AI...")
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "ui/streamlit_app.py", 
                "--server.headless", "true"
            ])
        
        elif choice == "2":
            if enhanced_available:
                print("\n🌟 Launching Enhanced InsightForge AI...")
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", 
                    "ui/enhanced_streamlit_app.py", 
                    "--server.headless", "true"
                ])
            else:
                print("\n🔧 Installing Enhanced Features...")
                install_enhanced_features()
        
        elif choice == "3" and enhanced_available:
            print("\n🎛️ Opening System Configuration...")
            show_configuration_menu()
        
        elif choice == "4":
            print("👋 Goodbye!")
            
        else:
            print("❌ Invalid choice. Please try again.")
            main()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

def install_enhanced_features():
    """Install enhanced features dependencies."""
    
    print("📦 Installing enhanced dependencies...")
    print("This will install: tavily-python, wikipedia, langchainhub")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Installation cancelled.")
        return
    
    try:
        # Install enhanced requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "tavily-python", "wikipedia", "langchainhub", "langchain-experimental"
        ], check=True)
        
        print("✅ Enhanced features installed successfully!")
        print("Please restart the application to use enhanced features.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("Please install manually:")
        print("  pip install tavily-python wikipedia langchainhub langchain-experimental")

def show_configuration_menu():
    """Show configuration options."""
    
    print("\n🎛️ System Configuration")
    print("=" * 40)
    print("1. 🔑 Setup API Keys")
    print("2. 🗃️ Rebuild Knowledge Base")
    print("3. 🧪 Test System Components")
    print("4. 📊 View System Status")
    print("5. 🔙 Back to Main Menu")
    
    config_choice = input("Choose option (1-5): ").strip()
    
    if config_choice == "1":
        setup_api_keys()
    elif config_choice == "2":
        rebuild_knowledge_base()
    elif config_choice == "3":
        test_components()
    elif config_choice == "4":
        show_system_status()
    elif config_choice == "5":
        main()
    else:
        print("❌ Invalid choice.")
        show_configuration_menu()

def setup_api_keys():
    """Interactive API key setup."""
    
    print("\n🔑 API Key Setup")
    print("=" * 30)
    
    # Check current .env file
    env_path = Path(".env")
    env_template_path = Path(".env.template")
    
    if not env_path.exists() and env_template_path.exists():
        print("📋 Creating .env file from template...")
        import shutil
        shutil.copy(env_template_path, env_path)
    
    print("Configure your API keys in the .env file:")
    print(f"📁 File location: {env_path.absolute()}")
    print()
    print("Available API keys to configure:")
    print("- OPENAI_API_KEY (for GPT models)")
    print("- ANTHROPIC_API_KEY (for Claude models)")
    print("- TAVILY_API_KEY (for web search)")
    print()
    print("💡 You can also configure these in the Streamlit interface")
    
    input("Press Enter to continue...")

def rebuild_knowledge_base():
    """Rebuild the knowledge base."""
    
    print("\n🗃️ Rebuilding Knowledge Base...")
    
    try:
        os.chdir("loaders")
        subprocess.run([sys.executable, "setup_knowledge_base.py"], check=True)
        os.chdir("..")
        print("✅ Knowledge base rebuilt successfully!")
    except Exception as e:
        print(f"❌ Error rebuilding knowledge base: {e}")
    
    input("Press Enter to continue...")

def test_components():
    """Test system components."""
    
    print("\n🧪 Testing System Components...")
    print("=" * 40)
    
    # Test knowledge base
    print("📚 Testing knowledge base...")
    if check_knowledge_base():
        print("  ✅ Knowledge base found")
    else:
        print("  ❌ Knowledge base missing")
    
    # Test dependencies
    print("📦 Testing dependencies...")
    
    # Core dependencies
    try:
        import streamlit, pandas, langchain
        print("  ✅ Core dependencies installed")
    except ImportError as e:
        print(f"  ❌ Core dependency missing: {e}")
    
    # Enhanced dependencies
    if check_dependencies():
        print("  ✅ Enhanced dependencies installed")
        
        # Test API connections
        print("🌐 Testing API connections...")
        test_api_connections()
    else:
        print("  ⚠️ Enhanced dependencies not installed")
    
    input("Press Enter to continue...")

def test_api_connections():
    """Test API connections if keys are available."""
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test Tavily
    tavily_key = os.getenv('TAVILY_API_KEY')
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            # Simple test search
            result = client.search("test", max_results=1)
            print("  ✅ Tavily API working")
        except Exception as e:
            print(f"  ❌ Tavily API error: {e}")
    else:
        print("  ⚠️ Tavily API key not configured")
    
    # Test Wikipedia
    try:
        import wikipedia
        result = wikipedia.summary("artificial intelligence", sentences=1)
        print("  ✅ Wikipedia API working")
    except Exception as e:
        print(f"  ❌ Wikipedia API error: {e}")

def show_system_status():
    """Show comprehensive system status."""
    
    print("\n📊 System Status")
    print("=" * 30)
    
    # Knowledge base status
    print("📚 Knowledge Base:")
    kb_path = Path("data/knowledge_base/faiss_index")
    if kb_path.exists():
        # Get file count and size
        file_count = len(list(kb_path.glob("*")))
        print(f"  ✅ Found ({file_count} files)")
    else:
        print("  ❌ Not found")
    
    # Data files status
    print("\n📊 Data Files:")
    data_path = Path("data")
    if (data_path / "sales_data.csv").exists():
        print("  ✅ sales_data.csv found")
    else:
        print("  ⚠️ sales_data.csv not found")
    
    pdf_path = data_path / "pdfs"
    if pdf_path.exists():
        pdf_count = len(list(pdf_path.glob("*.pdf")))
        print(f"  📄 {pdf_count} PDF files in pdfs/")
    else:
        print("  📄 No PDFs directory")
    
    # Dependencies status
    print("\n📦 Dependencies:")
    try:
        import streamlit
        print(f"  ✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("  ❌ Streamlit not installed")
    
    if check_dependencies():
        print("  ✅ Enhanced features available")
    else:
        print("  ⚠️ Enhanced features not installed")
    
    # Environment variables
    print("\n🔑 Environment Variables:")
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY')
    }
    
    for key, value in api_keys.items():
        if value:
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  ✅ {key}: {masked_value}")
        else:
            print(f"  ⚠️ {key}: Not set")
    
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()