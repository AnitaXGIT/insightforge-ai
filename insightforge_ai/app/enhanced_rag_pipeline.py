# app/enhanced_rag_pipeline.py - Inheritance Approach

import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
import warnings
import time
from datetime import datetime

# Import the working base pipeline
from app.rag_pipeline import (
    InsightForgeRAGPipeline,
    create_rag_pipeline_with_openai,
    create_rag_pipeline_with_anthropic,
    create_rag_pipeline_with_local_llm
)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


class EnhancedInsightForgeRAGPipeline(InsightForgeRAGPipeline):
    """
    Enhanced RAG Pipeline with external source fallback capabilities.
    Inherits all working functionality from InsightForgeRAGPipeline.
    """
    
    def __init__(
        self,
        knowledge_base_path: str = None,
        llm=None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_memory: bool = True,
        tavily_api_key: str = None,
        enable_fallback: bool = True,
        fallback_threshold: float = 0.3
    ):
        """
        Initialize enhanced pipeline by inheriting from working base.
        
        Args:
            All the same args as InsightForgeRAGPipeline, plus:
            tavily_api_key: API key for Tavily web search
            enable_fallback: Whether to enable external source fallback
            fallback_threshold: Confidence threshold for triggering fallback
        """
        
        # Initialize the working base class first
        print("🚀 Initializing Enhanced InsightForge RAG Pipeline (Inheritance)...")
        super().__init__(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            embedding_model=embedding_model,
            enable_memory=enable_memory
        )
        
        # Add enhanced features
        self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
        self.enable_fallback = enable_fallback
        self.fallback_threshold = fallback_threshold
        
        # External tools
        self.tavily_search = None
        self.wikipedia_tool = None
        self.agent_executor = None
        
        # Initialize external capabilities
        if self.enable_fallback:
            self._initialize_external_tools()
        
        print("✅ Enhanced RAG Pipeline ready with external capabilities!")
    

    def debug_knowledge_base(self):
        """Debug what's in the knowledge base."""
        if self.vector_store:
            print("🔍 Debugging knowledge base contents...")
            
            # Test different searches
            test_queries = ["sales", "AI", "artificial intelligence", "business", "strategy", "innovation"]
            
            for query in test_queries:
                docs = self.vector_store.similarity_search(query, k=3)
                print(f"\nQuery: '{query}' - Found {len(docs)} docs")
                for i, doc in enumerate(docs):
                    metadata = doc.metadata
                    preview = doc.page_content[:100] + "..."
                    source_type = "PDF" if metadata.get('source_type') == 'pdf' else "Sales" if metadata.get('section') else "Other"
                    print(f"  Doc {i+1} ({source_type}): {preview}")
                    print(f"    Metadata: {metadata}")
        else:
            print("❌ No vector store available")



    def _initialize_external_tools(self):
        """Initialize external search tools (Tavily, Wikipedia)."""
        print("🔧 Initializing external search tools...")
        
        try:
            # Initialize Tavily Search
            if self.tavily_api_key:
                self.tavily_search = TavilySearchResults(
                    api_key=self.tavily_api_key,
                    max_results=3,
                    search_depth="basic",
                    include_answer=True,
                    include_raw_content=False
                )
                print("✅ Tavily Search initialized")
            else:
                print("⚠️ Tavily API key not found - web search disabled")
            
            # Initialize Wikipedia
            try:
                wikipedia_wrapper = WikipediaAPIWrapper(
                    top_k_results=2,
                    doc_content_chars_max=2000
                )
                self.wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
                print("✅ Wikipedia tool initialized")
            except Exception as e:
                print(f"⚠️ Wikipedia initialization failed: {e}")
            
            # Setup agent tools
            self._setup_agent_tools()
            
        except Exception as e:
            print(f"❌ Error initializing external tools: {e}")
            # Continue without external tools
    
    def _setup_agent_tools(self):
        """Setup agent tools for external search."""
        if not self.llm:
            print("⚠️ No LLM available for agent - external search limited")
            return
        
        tools = []
        
        # Add Tavily search tool
        if self.tavily_search:
            tavily_tool = Tool(
                name="tavily_search",
                description="Search the web for current information, news, recent developments, and real-time data. Use this for questions about recent events, current statistics, or information not available in internal documents.",
                func=self._tavily_search_wrapper
            )
            tools.append(tavily_tool)
        
        # Add Wikipedia tool
        if self.wikipedia_tool:
            wiki_tool = Tool(
                name="wikipedia_search",
                description="Search Wikipedia for general knowledge, historical information, definitions, and encyclopedic content. Use this for questions about general concepts, historical events, or well-established facts.",
                func=self._wikipedia_search_wrapper
            )
            tools.append(wiki_tool)
        
        # Setup agent if we have tools
        if tools and self.llm:
            try:
                # Get the react prompt template
                prompt = hub.pull("hwchase17/react")
                
                # Create ReAct agent
                agent = create_react_agent(
                    llm=self.llm,
                    tools=tools,
                    prompt=prompt
                )
                
                self.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=3
                )
                
                print("✅ External search agent initialized")
                
            except Exception as e:
                print(f"⚠️ Agent setup failed: {e}")
    
    def _tavily_search_wrapper(self, query: str) -> str:
        """Wrapper for Tavily search tool."""
        try:
            results = self.tavily_search.run(query)
            if isinstance(results, list):
                formatted_results = []
                for result in results[:2]:  # Limit to top 2 results
                    if isinstance(result, dict):
                        title = result.get('title', 'No title')
                        content = result.get('content', result.get('snippet', 'No content'))
                        url = result.get('url', 'No URL')
                        formatted_results.append(f"**{title}**\n{content}\nSource: {url}")
                return "\n\n".join(formatted_results)
            return str(results)
        except Exception as e:
            return f"Tavily search error: {str(e)}"
    
    def _wikipedia_search_wrapper(self, query: str) -> str:
        """Wrapper for Wikipedia search tool."""
        try:
            result = self.wikipedia_tool.run(query)
            return result
        except Exception as e:
            return f"Wikipedia search error: {str(e)}"
    


    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """Enhanced query - works like original + adds Tavily search when needed."""
        print(f"\n🔍 Enhanced query: {question}")
        
        try:
            start_time = time.time()
            
            # Step 1: Use parent method (works exactly like original)
            print("📊 Getting internal response...")
            internal_result = super().query(question, return_sources)
            
            # Step 2: Check if we should add external search
            question_lower = question.lower()
            needs_external = any(word in question_lower for word in [
                'current', 'recent', 'latest', 'trends', 'what is', 'ai', 'artificial intelligence'
            ])
            
            # Step 3: Add Tavily search if needed and available
            if needs_external and self.tavily_search:
                print("🌐 Adding Tavily search...")
                try:
                    external_info = self._tavily_search_wrapper(question)
                    if external_info and len(external_info) > 50:  # Got good results
                        # Simple combination
                        combined_answer = f"{internal_result.get('answer', '')}\n\n**Additional Current Information:**\n{external_info}"
                        internal_result["answer"] = combined_answer
                        internal_result["used_fallback"] = True
                        print("✅ Enhanced with Tavily results")
                    else:
                        internal_result["used_fallback"] = False
                except Exception as e:
                    print(f"⚠️ Tavily search failed: {e}")
                    internal_result["used_fallback"] = False
            else:
                internal_result["used_fallback"] = False
            
            # Step 4: Add timing
            end_time = time.time()
            internal_result["response_time"] = end_time - start_time
            
            print(f"✅ Enhanced query completed in {internal_result['response_time']:.2f}s")
            return internal_result
            
        except Exception as e:
            print(f"❌ Enhanced query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "used_fallback": False,
                "error": str(e)
            }







    def _needs_external_enhancement(self, question: str, answer: str, sources: List) -> Tuple[bool, str]:
        """
        Determine if external enhancement is needed.
        More conservative approach - only enhance when clearly beneficial.
        """
        
        if not self.enable_fallback:
            return False, "External fallback disabled"
        
        # Check confidence in internal response
        confidence = self._assess_internal_confidence(answer, sources)
        
        if confidence < self.fallback_threshold:
            return True, f"Low internal confidence: {confidence:.3f} < {self.fallback_threshold}"
        
        # Check for explicit external request indicators
        external_indicators = [
            'current', 'recent', 'latest', 'today', 'now', '2024', '2025',
            'news', 'market trends', 'industry trends', 'what is happening',
            'what are the trends', 'current best practices'
        ]
        
        question_lower = question.lower()
        for indicator in external_indicators:
            if indicator in question_lower:
                return True, f"Question requests current/external information: '{indicator}'"
        
        # Check for general knowledge questions that might benefit from external sources
        knowledge_indicators = [
            'what is', 'define', 'explain', 'how does', 'what are the benefits',
            'best practices', 'artificial intelligence', 'ai', 'machine learning'
        ]
        
        for indicator in knowledge_indicators:
            if indicator in question_lower and confidence < 0.7:
                return True, f"General knowledge question with medium confidence: '{indicator}'"
        
        return False, "Internal response is sufficient"
    
    def _assess_internal_confidence(self, answer: str, sources: List) -> float:
        """Assess confidence in the internal response."""
        
        if not answer or len(answer.strip()) < 20:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Boost for business data indicators
        business_indicators = ['$', '%', 'sales', 'revenue', 'customers', 'products', 'data shows']
        for indicator in business_indicators:
            if indicator in answer.lower():
                confidence += 0.1
        
        # Boost for having good sources
        if sources and len(sources) > 0:
            confidence += 0.2
        
        # Reduce for uncertainty phrases
        uncertainty_phrases = [
            'i don\'t have', 'no information', 'cannot find', 'not available',
            'no specific data', 'unable to provide'
        ]
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                confidence -= 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def _get_external_information(self, question: str) -> Optional[str]:
        """Get information from external sources."""
        
        try:
            # Determine search strategy
            strategy = self._determine_search_strategy(question)
            
            if strategy == 'web' and self.tavily_search:
                return self._search_web(question)
            elif strategy == 'wikipedia' and self.wikipedia_tool:
                return self._search_wikipedia(question)
            elif strategy == 'both':
                web_info = self._search_web(question) if self.tavily_search else None
                wiki_info = self._search_wikipedia(question) if self.wikipedia_tool else None
                
                if web_info and wiki_info:
                    return f"**Web Search Results:**\n{web_info}\n\n**Wikipedia Information:**\n{wiki_info}"
                elif web_info:
                    return web_info
                elif wiki_info:
                    return wiki_info
            
            return None
            
        except Exception as e:
            print(f"❌ External search failed: {e}")
            return None
    
    def _determine_search_strategy(self, question: str) -> str:
        """Determine which external sources to use."""
        
        question_lower = question.lower()
        
        # Current/recent information -> web search
        if any(word in question_lower for word in ['current', 'recent', 'latest', 'today', 'trends']):
            return 'web'
        
        # General knowledge -> Wikipedia
        if any(word in question_lower for word in ['what is', 'define', 'explain', 'history of']):
            return 'wikipedia'
        
        # Business/AI topics -> both
        if any(word in question_lower for word in ['artificial intelligence', 'ai', 'business', 'strategy']):
            return 'both'
        
        return 'web'  # Default to web search
    
    def _search_web(self, question: str) -> Optional[str]:
        """Search the web using Tavily."""
        try:
            return self._tavily_search_wrapper(question)
        except Exception as e:
            print(f"Web search failed: {e}")
            return None
    
    def _search_wikipedia(self, question: str) -> Optional[str]:
        """Search Wikipedia."""
        try:
            return self._wikipedia_search_wrapper(question)
        except Exception as e:
            print(f"Wikipedia search failed: {e}")
            return None
    
    def _combine_internal_and_external(self, internal_result: Dict, external_info: str, question: str) -> Dict:
        """Combine internal and external information intelligently."""
        
        if not self.llm:
            # Simple combination if no LLM
            combined_answer = f"{internal_result.get('answer', '')}\n\n**Additional External Information:**\n{external_info}"
            internal_result["answer"] = combined_answer
            return internal_result
        
        # Use LLM to intelligently synthesize
        synthesis_prompt = f"""You are InsightForge AI. You have both internal business data and external information to answer this question.

Question: {question}

Internal Business Response:
{internal_result.get('answer', 'No internal information available')}

External Information:
{external_info}

Task: Create a comprehensive response that:
1. Prioritizes internal business data when available and relevant
2. Uses external information to provide additional context or fill gaps
3. Clearly indicates what comes from internal vs external sources
4. Provides actionable insights combining both sources
5. Maintains focus on business intelligence

Combined Response:"""

        try:
            if hasattr(self.llm, 'invoke'):
                result = self.llm.invoke(synthesis_prompt)
                synthesized_answer = result.content if hasattr(result, 'content') else str(result)
            else:
                synthesized_answer = self.llm(synthesis_prompt)
            
            # Update the result
            enhanced_result = internal_result.copy()
            enhanced_result["answer"] = synthesized_answer
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            # Fallback to simple combination
            combined_answer = f"{internal_result.get('answer', '')}\n\n**Additional Context from External Sources:**\n{external_info}"
            internal_result["answer"] = combined_answer
            return internal_result
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get status of fallback capabilities."""
        return {
            "fallback_enabled": self.enable_fallback,
            "tavily_available": self.tavily_search is not None,
            "wikipedia_available": self.wikipedia_tool is not None,
            "agent_available": self.agent_executor is not None,
            "fallback_threshold": self.fallback_threshold,
            "base_pipeline_ready": self.vector_store is not None
        }
    
    def set_fallback_threshold(self, threshold: float):
        """Update the fallback confidence threshold."""
        self.fallback_threshold = max(0.0, min(1.0, threshold))
        print(f"✅ Fallback threshold updated to {self.fallback_threshold}")
    
    def test_fallback_tools(self, test_query: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Test fallback tools functionality."""
        results = {
            "test_query": test_query,
            "tavily_result": None,
            "wikipedia_result": None,
            "base_pipeline_result": None
        }
        
        # Test base pipeline
        try:
            base_result = super().query(test_query, return_sources=True)
            results["base_pipeline_result"] = base_result.get("answer", "")[:200] + "..."
            print("✅ Base pipeline test successful")
        except Exception as e:
            results["base_pipeline_result"] = f"Error: {e}"
            print(f"❌ Base pipeline test failed: {e}")
        
        # Test Tavily
        if self.tavily_search:
            try:
                results["tavily_result"] = self._tavily_search_wrapper(test_query)
                print("✅ Tavily test successful")
            except Exception as e:
                results["tavily_result"] = f"Error: {e}"
                print(f"❌ Tavily test failed: {e}")
        
        # Test Wikipedia
        if self.wikipedia_tool:
            try:
                results["wikipedia_result"] = self._wikipedia_search_wrapper(test_query)
                print("✅ Wikipedia test successful")
            except Exception as e:
                results["wikipedia_result"] = f"Error: {e}"
                print(f"❌ Wikipedia test failed: {e}")
        
        return results


# Enhanced factory functions that use the working base factories

def create_enhanced_rag_with_openai(
    api_key: str, 
    knowledge_base_path: str = None,
    tavily_api_key: str = None,
    enable_fallback: bool = True
) -> EnhancedInsightForgeRAGPipeline:
    """Create enhanced RAG pipeline with OpenAI LLM using inheritance."""
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        return EnhancedInsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            tavily_api_key=tavily_api_key,
            enable_fallback=enable_fallback,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install langchain-openai")


def create_enhanced_rag_with_anthropic(
    api_key: str, 
    knowledge_base_path: str = None,
    tavily_api_key: str = None,
    enable_fallback: bool = True
) -> EnhancedInsightForgeRAGPipeline:
    """Create enhanced RAG pipeline with Anthropic Claude LLM using inheritance."""
    
    try:
        from langchain_anthropic import ChatAnthropic
        
        llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            temperature=0.1
        )
        
        return EnhancedInsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            tavily_api_key=tavily_api_key,
            enable_fallback=enable_fallback,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install langchain-anthropic")


def create_enhanced_rag_with_local_llm(
    model_path: str, 
    knowledge_base_path: str = None,
    tavily_api_key: str = None,
    enable_fallback: bool = True
) -> EnhancedInsightForgeRAGPipeline:
    """Create enhanced RAG pipeline with local LLM using inheritance."""
    
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(
            model=model_path,
            temperature=0.1
        )
        
        return EnhancedInsightForgeRAGPipeline(
            knowledge_base_path=knowledge_base_path,
            llm=llm,
            tavily_api_key=tavily_api_key,
            enable_fallback=enable_fallback,
            enable_memory=True
        )
    except ImportError:
        raise ImportError("Ollama package not installed. Run: pip install langchain-community")