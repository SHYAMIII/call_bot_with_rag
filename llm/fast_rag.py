import os
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from groq import Groq
import time
import threading
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

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
            temperature=0.1,  # Lower temperature for faster, more consistent responses
            max_tokens=150  # Limit tokens for faster generation
        )
        
        # Initialize TF-IDF vectorizer (optimized for speed)
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced from 1000 for faster processing
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for speed
            min_df=1,
            max_df=0.95
        )
        
        self.documents = []
        self.tfidf_matrix = None
        
        # TTL Cache for responses (5 minutes)
        self.response_cache = TTLCache(maxsize=200, ttl=300)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[str]):
        """Add documents to the TF-IDF index."""
        if not documents:
            return
            
        self.documents = documents
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        logging.info(f"Added {len(documents)} documents to TF-IDF index")
    
    @lru_cache(maxsize=1000)
    def _cached_retrieve(self, query: str, top_k: int = 3) -> tuple:
        """Cached retrieval for repeated queries."""
        if not self.tfidf_matrix is not None or not self.documents:
            return ()
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return relevant documents with scores
        retrieved_docs = [(self.documents[i], similarities[i]) for i in top_indices if similarities[i] > 0.01]
        
        # If no documents found with threshold, return at least the top document
        if not retrieved_docs and len(top_indices) > 0:
            retrieved_docs = [(self.documents[top_indices[0]], similarities[top_indices[0]])]
            
        return tuple(retrieved_docs)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Fast retrieval using TF-IDF and cosine similarity with caching."""
        cached_result = self._cached_retrieve(query, top_k)
        return [doc for doc, score in cached_result]
    
    async def retrieve_async(self, query: str, top_k: int = 3) -> List[str]:
        """Async retrieval for non-blocking operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.retrieve, query, top_k)
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Groq LLM with caching."""
        if not context:
            return "I don't have enough information to answer that question."
        
        # Check cache first
        cache_key = f"{query}:{hash(tuple(context))}"
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Combine context
        context_text = "\n".join(context)
        
        # Optimized prompt for faster generation
        prompt = f"""Answer based only on the context. Be concise and direct.

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