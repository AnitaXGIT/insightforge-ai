# app/rag_pipeline.py

import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import warnings
from app.query_router import InsightForgeQueryRouter

# Suppress LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


class InsightForgeRAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) Pipeline for InsightForge AI.
    Combines knowledge base retrieval with LLM generation for intelligent business insights.
    """
    
    def __init__(
        self, 
        knowledge_base_path: str = None,
        llm: Optional[LLM] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_memory: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            knowledge_base_path: Path to the saved knowledge base
            llm: Language model instance (OpenAI, Anthropic, local, etc.)
            embedding_model: HuggingFace embedding model name
            enable_memory: Whether to enable conversation memory
        """
        self.knowledge_base_path = knowledge_base_path or self._get_default_kb_path()
        self.llm = llm
        self.embedding_model = embedding_model
        self.enable_memory = enable_memory
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.memory = None
        self.qa_chain = None
        self.query_router = None
        
        # Initialize the pipeline
        self._initialize_pipeline()
    


    def _get_default_kb_path(self) -> str:
        """Get default knowledge base path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(current_dir), "data", "knowledge_base")
    


    def _initialize_pipeline(self):
        """Initialize all components of the RAG pipeline."""
        print("Initializing InsightForge RAG Pipeline...")
        
        try:
            # 1. Initialize embeddings
            self._initialize_embeddings()
            
            # 2. Load knowledge base
            self._load_knowledge_base()
            
            # 3. Setup retriever
            self._setup_retriever()
            
            # 4. Initialize memory (if enabled)
            if self.enable_memory:
                self._initialize_memory()
            
            # 5. Setup QA chain
            self._setup_qa_chain()

            # 6. Initialize query router
            self._setup_query_router()
            
            print("✅ RAG Pipeline initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
            raise
    
    def _setup_query_router(self):
        """Setup the intelligent query router."""
        self.query_router = InsightForgeQueryRouter()
        print("✅ Query router initialized")


    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        print("✅ Embeddings loaded")
    


    def _load_knowledge_base(self):
        """Load the pre-built knowledge base."""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base not found at: {self.knowledge_base_path}")
        
        faiss_path = os.path.join(self.knowledge_base_path, "faiss_index")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at: {faiss_path}")
        
        print(f"Loading knowledge base from: {self.knowledge_base_path}")

        self.vector_store = FAISS.load_local(
        faiss_path, 
        self.embeddings, 
        allow_dangerous_deserialization=True     #Add allow_dangerous_deserialization=True since we trust our own files
        )

        print(f"✅ Knowledge base loaded with {self.vector_store.index.ntotal} vectors")
    


    def _setup_retriever(self, k: int = 5, search_type: str = "similarity"):
        """
        Setup the retriever with customizable parameters.
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
        """
        if search_type == "similarity":
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        elif search_type == "mmr":
            # Maximal Marginal Relevance for diversity
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k*2}
            )
        elif search_type == "similarity_score_threshold":
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.7, "k": k}
            )
        
        print(f"✅ Retriever setup with {search_type} search, k={k}")
    


    def _initialize_memory(self):
        """Initialize conversation memory for context retention."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        print("✅ Conversation memory initialized")
    


    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompts."""
        if self.llm is None:
            print("⚠ No LLM provided. Using a mock LLM for testing.")
            self.llm = self._create_mock_llm()
        
        # Create custom prompt template for business intelligence
        prompt_template = self._create_bi_prompt_template()
        
        if self.enable_memory and self.memory:
            # Use ConversationalRetrievalChain for memory-enabled conversations
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )
        else:
            # Use RetrievalQA for stateless queries
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
        
        print("✅ QA chain setup complete")
    


    def _create_bi_prompt_template(self) -> PromptTemplate:
        """Create a specialized prompt template for business intelligence queries."""
        
        template = """You are InsightForge AI, an expert Business Intelligence assistant. You analyze business data and provide actionable insights.

Context Information:
{context}

Instructions:
- Provide clear, data-driven insights based on the context
- Focus on actionable business recommendations
- Use specific numbers and metrics when available
- Organize your response with clear headings when appropriate
- If the context contains both sales data and external documents, synthesize insights from both sources
- Always cite which source your information comes from (sales data, PDF documents, etc.)

Question: {question}

Business Intelligence Response:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    


    def _create_mock_llm(self):
        """Create a mock LLM for testing when no real LLM is provided."""
        
        class MockLLM(LLM):
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return "This is a mock response. Please configure a real LLM (OpenAI, Anthropic, etc.) for full functionality."
            
            @property
            def _llm_type(self) -> str:
                return "mock"
        
        return MockLLM()
    

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline with intelligent routing and debug info."""
        
        if not self.qa_chain:
            raise ValueError("QA chain not initialized.")
        
        print(f"\n🔍 Processing query: {question}")
        
        try:
            # Use custom retrieval with query routing
            if self.query_router and self.vector_store:
                print("🚀 Using intelligent query routing")
                
                # Get routed context documents
                context_docs = self.query_router.route_retrieval(question, self.vector_store, k=5)
                print(f"📊 Retrieved {len(context_docs)} documents after routing")
                
                # Debug: Show what we actually got
                if context_docs:
                    for i, doc in enumerate(context_docs):
                        metadata = doc.metadata
                        content_preview = doc.page_content[:100] + "..."
                        print(f"   Doc {i+1}: {metadata} | {content_preview}")
                else:
                    print("⚠️ No documents retrieved from routing")
                
                # Create context string from routed documents
                if context_docs:
                    context_text = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    # Use LLM directly with routed context
                    if self.llm:
                        prompt = f"""You are InsightForge AI, a business intelligence assistant.

Context Information:
{context_text}

Question: {question}

Please provide insights based on the context above. If the context contains sales data, focus on specific metrics and numbers. If it contains strategic information, provide recommendations and insights."""

                        try:
                            # Handle different LLM types
                            if hasattr(self.llm, 'invoke'):
                                # For ChatModels (OpenAI, Anthropic)
                                result = self.llm.invoke(prompt)
                                if hasattr(result, 'content'):
                                    answer = result.content
                                else:
                                    answer = str(result)
                            else:
                                # For regular LLMs
                                answer = self.llm(prompt)
                        except Exception as e:
                            print(f"❌ LLM error: {e}")
                            answer = f"Error generating response: {str(e)}"
                    else:
                        answer = "Mock response: Please configure a real LLM for full functionality."
                else:
                    # No documents found
                    answer = "I don't have specific data to answer your question. Please check if your knowledge base contains relevant information."
                
                # Format response with debug info
                formatted_response = {
                    "question": question,
                    "answer": answer,
                    "source_count": len(context_docs),
                    "routing_debug": {
                        "query_type": self.query_router.classify_query(question) if self.query_router else "unknown",
                        "documents_found": len(context_docs),
                        "source_types": [doc.metadata.get('source', 'unknown') for doc in context_docs]
                    }
                }
                
                if return_sources:
                    formatted_response["sources"] = self._format_sources(context_docs)
                
                return formatted_response
                
            else:
                print("⚠️ No query router available - using fallback method")
                # Fallback to original method
                if self.enable_memory:
                    result = self.qa_chain({"question": question})
                    answer = result.get("answer", "")
                    sources = result.get("source_documents", []) if return_sources else []
                else:
                    result = self.qa_chain({"query": question})
                    answer = result.get("result", "")
                    sources = result.get("source_documents", []) if return_sources else []
                
                formatted_response = {
                    "question": question,
                    "answer": answer,
                    "source_count": len(sources)
                }
                
                if return_sources:
                    formatted_response["sources"] = self._format_sources(sources)
                
                return formatted_response
                
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "source_count": 0,
                "sources": [],
                "error": str(e)
            }
    




    def _format_sources(self, sources: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for better readability."""
        formatted_sources = []
        
        for i, doc in enumerate(sources):
            source_info = {
                "index": i + 1,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            
            # Add source type information
            metadata = doc.metadata
            if metadata.get('source_type') == 'pdf':
                source_info['source_type'] = f"PDF: {metadata.get('source_file', 'Unknown')}"
            elif metadata.get('section'):
                source_info['source_type'] = f"Sales Data: {metadata['section']}"
            else:
                source_info['source_type'] = "Sales Summary"
            
            formatted_sources.append(source_info)
        
        return formatted_sources
    


    def get_relevant_context(self, question: str, k: int = 3) -> List[Document]:
        """
        Get relevant context documents with intelligent routing.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized.")
    
        # Use query router if available, otherwise fall back to normal retrieval
        if self.query_router and self.vector_store:
            return self.query_router.route_retrieval(question, self.vector_store, k)
        else:
            return self.retriever.get_relevant_documents(question)
    


    def update_retriever_config(self, k: int = 5, search_type: str = "similarity"):
        """Update retriever configuration dynamically."""
        self._setup_retriever(k=k, search_type=search_type)
        
        # Recreate QA chain with updated retriever
        self._setup_qa_chain()
        print(f"✅ Retriever updated: {search_type}, k={k}")
    


    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            print("✅ Conversation memory cleared")
    


    def get_memory_summary(self) -> str:
        """Get a summary of the conversation memory."""
        if not self.memory:
            return "Memory not enabled"
        
        try:
            return str(self.memory.buffer)
        except:
            return "Memory is empty"




def create_rag_pipeline_with_openai(api_key: str, knowledge_base_path: str = None) -> InsightForgeRAGPipeline:
    """
    Create RAG pipeline with OpenAI LLM.
    
    Args:
        api_key: OpenAI API key
        knowledge_base_path: Path to knowledge base
        
    Returns:
        Configured RAG pipeline
    """
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        return InsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install langchain-openai")




def create_rag_pipeline_with_anthropic(api_key: str, knowledge_base_path: str = None) -> InsightForgeRAGPipeline:
    """
    Create RAG pipeline with Anthropic Claude LLM.
    
    Args:
        api_key: Anthropic API key
        knowledge_base_path: Path to knowledge base
        
    Returns:
        Configured RAG pipeline
    """
    try:
        from langchain_anthropic import ChatAnthropic
        
        llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            temperature=0.1
        )
        
        return InsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install langchain-anthropic")




def create_rag_pipeline_with_local_llm(model_path: str, knowledge_base_path: str = None) -> InsightForgeRAGPipeline:
    """
    Create RAG pipeline with local LLM (e.g., Ollama, LlamaCpp).
    
    Args:
        model_path: Path or name of local model
        knowledge_base_path: Path to knowledge base
        
    Returns:
        Configured RAG pipeline
    """
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(
            model=model_path,
            temperature=0.1
        )
        
        return InsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("Ollama package not installed. Run: pip install langchain-community")

