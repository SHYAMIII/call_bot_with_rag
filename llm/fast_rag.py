import os
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
import time
import threading
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

class TTLCache:
    """Simple TTL cache for responses."""
    def __init__(self, maxsize=100, ttl=300):  # 5 minutes TTL
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove oldest
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                    del self.timestamps[oldest]
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        with self.lock:
            expired_keys = [k for k, v in self.timestamps.items() 
                          if current_time - v >= self.ttl]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]

class FastRAG:
    def __init__(self, documents: List[str] = None):
        """
        Initialize FastRAG with documents for real-time retrieval.
        
        Args:
            documents: List of text documents to index
        """
        # Initialize LLM (Groq - fastest API with optimized model)
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY, 
            model_name="llama3-8b-8192",  # Faster model than 70b
            temperature=0.4,  # Lower temperature for faster, more consistent responses
            max_tokens=150  # Limit tokens for faster generation
        )
        
        # Initialize embedding model and FAISS index
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.faiss_index = None
        self.documents = []
        self.doc_embeddings = None
        
        # TTL Cache for responses (5 minutes)
        self.response_cache = TTLCache(maxsize=200, ttl=300)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[str]):
        """Add documents to the FAISS index."""
        if not documents:
            return
        self.documents = documents
        # Generate dense embeddings
        self.doc_embeddings = self.embedder.encode(documents, show_progress_bar=False, convert_to_numpy=True)
        # Build FAISS index
        dim = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self.doc_embeddings)
        logging.info(f"Added {len(documents)} documents to FAISS index")
    
    @lru_cache(maxsize=1000)
    def _cached_retrieve(self, query: str, top_k: int = 3) -> tuple:
        """Cached retrieval for repeated queries using FAISS."""
        if self.faiss_index is None or not self.documents:
            return ()
        
        # Embed the query
        query_embedding = self.embedder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        
        # Search FAISS index
        D, I = self.faiss_index.search(query_embedding, top_k)
        
        # Return relevant documents with scores (distance)
        retrieved_docs = [(self.documents[i], float(D[0][rank])) for rank, i in enumerate(I[0]) if i < len(self.documents)]
        
        return tuple(retrieved_docs)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Fast retrieval using FAISS and dense embeddings with caching."""
        cached_result = self._cached_retrieve(query, top_k)
        return [doc for doc, score in cached_result]
    
    async def retrieve_async(self, query: str, top_k: int = 3) -> List[str]:
        """Async retrieval for non-blocking operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.retrieve, query, top_k)
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Groq LLM with caching."""
        if not context:
            return "I don't have enough information to answer that question. you can reach out to our team at info@technologymindz.com"
        
        # Check cache first
        cache_key = f"{query}:{hash(tuple(context))}"
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Combine context
        context_text = "\n".join(context)
        
        # Optimized prompt for faster generation
        prompt = f"""Answer based only on the context.dont greet the user. Be friendly and focused on the question.

Context: {context_text}
Question: {query}
Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            # Cache the response
            self.response_cache.set(cache_key, result)
            
            return result
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    async def generate_response_async(self, query: str, context: List[str]) -> str:
        """Async response generation for non-blocking operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_response, query, context)
    
    def query(self, question: str, top_k: int = 3, return_timings: bool = False):
        """Complete RAG pipeline: retrieve + generate with caching."""
        timings = {}
        t0 = time.time()
        
        # Clean expired cache entries periodically
        if hasattr(self, '_last_cache_cleanup'):
            if time.time() - self._last_cache_cleanup > 60:  # Clean every minute
                self.response_cache.clear_expired()
                self._last_cache_cleanup = time.time()
        else:
            self._last_cache_cleanup = time.time()
        
        relevant_docs = self.retrieve(question, top_k)
        t1 = time.time()
        response = self.generate_response(question, relevant_docs)
        t2 = time.time()
        if return_timings:
            timings = {
                "retrieval_time": t1 - t0,
                "generation_time": t2 - t1,
                "total_time": t2 - t0
            }
            return response, timings
        return response
    
    async def query_async(self, question: str, top_k: int = 3, return_timings: bool = False):
        """Async RAG pipeline for non-blocking operations."""
        timings = {}
        t0 = time.time()
        relevant_docs = await self.retrieve_async(question, top_k)
        t1 = time.time()
        response = await self.generate_response_async(question, relevant_docs)
        t2 = time.time()
        if return_timings:
            timings = {
                "retrieval_time": t1 - t0,
                "generation_time": t2 - t1,
                "total_time": t2 - t0
            }
            return response, timings
        return response

# Global instance for easy access (pre-loaded)
fast_rag = None
_initialization_lock = threading.Lock()

def initialize_fast_rag(documents: List[str] = None) -> FastRAG:
    """Initialize FastRAG with your documents."""
    if documents is None:
        # Load documents from JSON file
        try:
            from data.load_documents import load_company_documents
            documents = load_company_documents()
        except ImportError:
            # Fallback to example documents
            documents = [
                "Technology Mindz is a software development company specializing in AI and Salesforce solutions.",
                "We offer custom software development, AI integration, and Salesforce consulting services.",
                "Our team has expertise in Python, JavaScript, React, and various AI frameworks.",
                "We provide end-to-end solutions from concept to deployment and maintenance.",
                "Contact us for a free consultation about your software development needs."
            ]
    
    return FastRAG(documents)

def get_fast_rag_response(question: str, return_timings: bool = False):
    """Get response from FastRAG (for use in your existing routes). Optionally return timing info."""
    global fast_rag
    
    # Thread-safe initialization
    if fast_rag is None:
        with _initialization_lock:
            if fast_rag is None:
                fast_rag = initialize_fast_rag()
    
    if return_timings:
        return fast_rag.query(question, return_timings=True)
    return fast_rag.query(question)

async def get_fast_rag_response_async(question: str, return_timings: bool = False):
    """Async version of get_fast_rag_response for non-blocking operations."""
    global fast_rag
    
    # Thread-safe initialization
    if fast_rag is None:
        with _initialization_lock:
            if fast_rag is None:
                fast_rag = initialize_fast_rag()
    
    if return_timings:
        return await fast_rag.query_async(question, return_timings=True)
    return await fast_rag.query_async(question)

# Pre-load the RAG system on import
def preload_rag():
    """Pre-load the RAG system to reduce first-query latency."""
    global fast_rag
    if fast_rag is None:
        fast_rag = initialize_fast_rag()
        print("RAG system pre-loaded and ready!")

# Auto-preload when module is imported
preload_rag() 