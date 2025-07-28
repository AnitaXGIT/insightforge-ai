# app/query_router.py - STRICT Sales-Only Query Router Fix

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import re

class InsightForgeQueryRouter:
    """
    Routes queries to appropriate data sources based on question type.
    STRICTLY enforces sales-only responses for metrics questions.
    """
    
    def __init__(self):
        # Keywords that indicate business metrics questions
        self.metrics_keywords = {
            'sales', 'revenue', 'total', 'figures', 'numbers', 'performance',
            'metrics', 'kpi', 'results', 'amount', 'value', 'sum', 'average',
            'median', 'statistics', 'data', 'trends', 'growth', 'profit',
            'customers', 'products', 'regions', 'demographic', 'segment',
            'best', 'worst', 'compare', 'analysis', 'how much', 'what are'
        }
        
        # Keywords that indicate strategic/AI questions
        self.strategy_keywords = {
            'ai', 'artificial intelligence', 'strategy', 'strategic', 'recommendations',
            'improve', 'optimize', 'innovation', 'transform', 'implement',
            'framework', 'approach', 'methodology', 'best practices', 'how can',
            'what should', 'suggest', 'advise'
        }

    def classify_query(self, question: str) -> str:
        """Classify query with stricter metrics detection."""
        question_lower = question.lower()
        
        # More aggressive metrics detection - if ANY metrics keyword is found, classify as metrics
        for keyword in self.metrics_keywords:
            if keyword in question_lower:
                return 'metrics'
        
        # Otherwise check for strategy keywords
        for keyword in self.strategy_keywords:
            if keyword in question_lower:
                return 'strategy'
        
        return 'general'

    def route_retrieval(self, question: str, vector_store, k: int = 5) -> List[Document]:
        """
        Route retrieval based on question type with STRICT source filtering.
        """
        query_type = self.classify_query(question)
        print(f"🔍 Query type classified as: {query_type}")
        
        if query_type == 'metrics':
            return self._retrieve_sales_only_strict(question, vector_store, k)
        elif query_type == 'strategy':
            return self._retrieve_with_strategy_priority(question, vector_store, k)
        else:
            return self._retrieve_balanced(question, vector_store, k)

    def _retrieve_sales_only_strict(self, question: str, vector_store, k: int) -> List[Document]:
        """ABSOLUTELY STRICTLY use sales summary for metrics questions - NO PDF ALLOWED."""
        
        print(f"🎯 STRICT METRICS MODE: Only sales data allowed")
        
        # Get many documents to have better filtering options
        all_docs = vector_store.similarity_search(question, k=k*5)  # Get 5x more for filtering
        print(f"📊 Retrieved {len(all_docs)} total documents for STRICT filtering")
        
        # STRICT filtering - absolutely no PDFs
        sales_only_docs = []
        
        for i, doc in enumerate(all_docs):
            metadata = doc.metadata
            
            # STRICT CHECK: Absolutely reject anything that looks like PDF
            if self._is_pdf_document(metadata):
                print(f"❌ REJECTED PDF doc {i+1}: {metadata}")
                continue
            
            # STRICT CHECK: Only accept confirmed sales documents
            if self._is_sales_document(metadata):
                sales_only_docs.append(doc)
                print(f"✅ ACCEPTED sales doc {i+1}: {metadata.get('section', 'sales_summary')}")
            else:
                print(f"❓ REJECTED unknown doc {i+1}: {metadata}")
        
        print(f"📈 STRICT FILTERING RESULT: {len(sales_only_docs)} sales documents")
        
        if sales_only_docs:
            result = sales_only_docs[:k]
            print(f"🎯 RETURNING {len(result)} SALES-ONLY documents")
            return result
        else:
            print("⚠️ NO SALES DOCUMENTS FOUND - returning empty list")
            # Return empty list to force "no data available" response
            return []

    def _is_pdf_document(self, metadata: Dict) -> bool:
        """Check if document is from PDF source."""
        return (
            metadata.get('source_type') == 'pdf' or
            'pdf' in str(metadata.get('source_file', '')).lower() or
            'pdf' in str(metadata.get('file_path', '')).lower() or
            metadata.get('source_directory') is not None
        )

    def _is_sales_document(self, metadata: Dict) -> bool:
        """Check if document is confirmed sales data."""
        return (
            metadata.get('source') == 'sales_summary' or
            metadata.get('type') == 'sales_data' or
            metadata.get('section') is not None or  # Sales sections have section metadata
            'sales_summary' in str(metadata.get('source', '')).lower()
        )

    def _retrieve_with_strategy_priority(self, question: str, vector_store, k: int) -> List[Document]:
        """Prioritize PDF documents for strategy questions."""
        
        print(f"🧠 Strategy question detected - prioritizing PDF documents")
        
        # Get more documents than needed
        all_docs = vector_store.similarity_search(question, k=k*2)
        
        # Separate by source type
        pdf_docs = []
        sales_docs = []
        
        for doc in all_docs:
            if self._is_pdf_document(doc.metadata):
                pdf_docs.append(doc)
            else:
                sales_docs.append(doc)
        
        # Prioritize PDF docs, then fill with sales
        result = pdf_docs[:k//2 + 1]  # At least half from PDFs
        remaining = k - len(result)
        if remaining > 0:
            result.extend(sales_docs[:remaining])
        
        print(f"📄 Returning {len([d for d in result if self._is_pdf_document(d.metadata)])} PDF docs and {len([d for d in result if not self._is_pdf_document(d.metadata)])} sales docs")
        return result[:k]
    
    def _retrieve_balanced(self, question: str, vector_store, k: int) -> List[Document]:
        """Balanced retrieval for general questions."""
        print(f"⚖️ General question - using balanced retrieval")
        return vector_store.similarity_search(question, k=k)