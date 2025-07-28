# loaders/pdf_loader.py

import os
import pickle
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import Document as LangChainDocument
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import glob


class SalesKnowledgeBaseCreator:
    """
    Creates a knowledge base from sales summary text for RAG retrieval.
    Processes text into chunks, generates embeddings, and sets up vector storage.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the knowledge base creator.
        
        Args:
            embedding_model: HuggingFace model name for generating embeddings
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "–", "-", " ", ""]
        )
        self.vector_store = None




    def load_pdf_documents(self, pdf_paths: List[str] = None, pdf_directory: str = None) -> List[LangChainDocument]:
        """
        Load PDF documents from file paths or directory.
        
        Args:
            pdf_paths: List of specific PDF file paths to load
            pdf_directory: Directory containing PDF files to load
            
        Returns:
            List[LangChainDocument]: Loaded PDF documents
        """
        documents = []
        
        if pdf_paths:
            # Load specific PDF files
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    print(f"Warning: PDF file not found: {pdf_path}")
                    continue
                    
                try:
                    loader = PyPDFLoader(pdf_path)
                    pdf_docs = loader.load()
                    
                    # Add metadata about the source file
                    for doc in pdf_docs:
                        doc.metadata.update({
                            "source_file": os.path.basename(pdf_path),
                            "source_type": "pdf",
                            "file_path": pdf_path
                        })
                    
                    documents.extend(pdf_docs)
                    print(f"Loaded {len(pdf_docs)} pages from {os.path.basename(pdf_path)}")
                    
                except Exception as e:
                    print(f"Error loading {pdf_path}: {e}")
        
        if pdf_directory:
            # Load all PDFs from directory
            if not os.path.exists(pdf_directory):
                raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
            
            try:
                loader = DirectoryLoader(
                    pdf_directory,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    recursive=True
                )
                directory_docs = loader.load()
                
                # Add metadata
                for doc in directory_docs:
                    doc.metadata.update({
                        "source_type": "pdf",
                        "source_directory": pdf_directory
                    })
                
                documents.extend(directory_docs)
                print(f"Loaded {len(directory_docs)} pages from directory {pdf_directory}")
                
            except Exception as e:
                print(f"Error loading PDFs from directory {pdf_directory}: {e}")
        
        return documents
    



    def load_sales_summary(self, summary_path: str = None) -> str:
        """
        Load the sales summary text file.
        
        Args:
            summary_path: Path to the sales summary text file
            
        Returns:
            str: Content of the sales summary
        """
        if summary_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            summary_path = os.path.join(os.path.dirname(current_dir), "data", "sales_summary.txt")
        
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Sales summary file not found at: {summary_path}")
            
        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()
    



    def create_structured_documents(self, summary_text: str) -> List[LangChainDocument]:
        """
        Process sales summary into structured documents with metadata.
        
        Args:
            summary_text: Raw sales summary text
            
        Returns:
            List[LangChainDocument]: Processed documents with metadata
        """
        documents = []
        
        # Split the summary into logical sections
        sections = self._parse_summary_sections(summary_text)
        
        # Create documents for each section with appropriate metadata
        for section_name, content in sections.items():
            # Further split large sections into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = LangChainDocument(
                    page_content=chunk,
                    metadata={
                        "section": section_name,
                        "chunk_id": f"{section_name}_{i}",
                        "source": "sales_summary",
                        "type": "sales_data"
                    }
                )
                documents.append(doc)
        
        return documents
    



    def _parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        """
        Parse the sales summary into logical sections for better retrieval.
        
        Args:
            summary_text: Raw sales summary text
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to content
        """
        sections = {}
        lines = summary_text.strip().split('\n')
        
        current_section = "overview"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify section headers
            if any(keyword in line.lower() for keyword in ["time window", "product", "regions", "customers"]):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    current_content = []
                
                # Start new section
                if "time" in line.lower():
                    current_section = "temporal_analysis"
                elif "product" in line.lower():
                    current_section = "product_analysis"
                elif "region" in line.lower():
                    current_section = "regional_analysis"
                elif "customer" in line.lower():
                    current_section = "customer_analysis"
                    
                current_content.append(line)
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    



    def create_vector_store(self, documents: List[LangChainDocument]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            FAISS: Vector store for similarity search
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store
    



    def save_knowledge_base(self, save_path: str = None) -> str:
        """
        Save the vector store and metadata to disk.
        
        Args:
            save_path: Directory path to save the knowledge base
            
        Returns:
            str: Path where the knowledge base was saved
        """
        if save_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(os.path.dirname(current_dir), "data", "knowledge_base")
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.vector_store is None:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        
        # Save FAISS index
        faiss_path = os.path.join(save_path, "faiss_index")
        self.vector_store.save_local(faiss_path)
        
        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap,
            "creation_timestamp": None  # Could add timestamp if needed
        }
        
        metadata_path = os.path.join(save_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        return save_path
    



    def load_knowledge_base(self, load_path: str = None) -> FAISS:
        """
        Load a previously saved knowledge base.
        
        Args:
            load_path: Directory path to load the knowledge base from
            
        Returns:
            FAISS: Loaded vector store
        """
        if load_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            load_path = os.path.join(os.path.dirname(current_dir), "data", "knowledge_base")
        
        faiss_path = os.path.join(load_path, "faiss_index")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"Knowledge base not found at: {faiss_path}")
        
        # Add allow_dangerous_deserialization=True since we trust our own files
        self.vector_store = FAISS.load_local(
        faiss_path, 
        self.embeddings,
        allow_dangerous_deserialization=True
    )
        return self.vector_storeload_local(faiss_path, self.embeddings)

    

    def create_complete_knowledge_base(
        self, 
        summary_path: str = None, 
        save_path: str = None,
        pdf_paths: List[str] = None,
        pdf_directory: str = None,
        include_sales_summary: bool = True
    ) -> FAISS:
        """
        Complete pipeline: load summary and/or PDFs, create documents, build vector store, and save.
        
        Args:
            summary_path: Path to sales summary file
            save_path: Path to save knowledge base
            pdf_paths: List of specific PDF files to load
            pdf_directory: Directory containing PDF files
            include_sales_summary: Whether to include the sales summary text
            
        Returns:
            FAISS: Created vector store
        """
        all_documents = []
        
        # Load sales summary if requested
        if include_sales_summary:
            try:
                print("Loading sales summary...")
                summary_text = self.load_sales_summary(summary_path)
                
                print("Creating structured documents from sales summary...")
                summary_documents = self.create_structured_documents(summary_text)
                all_documents.extend(summary_documents)
                print(f"Created {len(summary_documents)} document chunks from sales summary")
            except Exception as e:
                print(f"Warning: Could not load sales summary: {e}")
        
        # Load PDF documents if specified
        if pdf_paths or pdf_directory:
            print("Loading PDF documents...")
            pdf_documents = self.load_pdf_documents(pdf_paths, pdf_directory)
            
            if pdf_documents:
                print("Processing PDF documents into chunks...")
                # Process PDFs with text splitter
                pdf_chunks = []
                for doc in pdf_documents:
                    chunks = self.text_splitter.split_documents([doc])
                    pdf_chunks.extend(chunks)
                
                all_documents.extend(pdf_chunks)
                print(f"Created {len(pdf_chunks)} document chunks from PDFs")
        
        if not all_documents:
            raise ValueError("No documents loaded. Please provide sales summary and/or PDF files.")
        
        print(f"Total documents: {len(all_documents)}")
        print("Building vector store...")
        vector_store = self.create_vector_store(all_documents)
        
        print("Saving knowledge base...")
        saved_path = self.save_knowledge_base(save_path)
        print(f"Knowledge base saved to: {saved_path}")
        
        return vector_store
    



    def test_retrieval(self, query: str, k: int = 3) -> List[LangChainDocument]:
        """
        Test the knowledge base retrieval with a sample query.
        
        Args:
            query: Test query string
            k: Number of documents to retrieve
            
        Returns:
            List[LangChainDocument]: Retrieved documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load knowledge base first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results)} documents:")
        print("-" * 50)
        
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Section: {doc.metadata.get('section', 'unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        return results

