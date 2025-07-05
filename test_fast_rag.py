#!/usr/bin/env python3
"""
Test script for FastRAG implementation
"""

from llm.fast_rag import initialize_fast_rag, get_fast_rag_response
import time

def test_fast_rag():
    """Test the FastRAG implementation."""
    print("Testing FastRAG implementation...")
    
    # Test questions
    test_questions = [
        "What services does Technology Mindz offer?",
        "Do you provide AI solutions?",
        "What technologies do you work with?",
        "How can I contact you?",
        "What is your expertise?"
    ]
    
    # Initialize RAG
    print("Initializing FastRAG...")
    rag = initialize_fast_rag()
    
    # Test responses
    for question in test_questions:
        print(f"\nQuestion: {question}")
        start_time = time.time()
        response = rag.query(question)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {(end_time - start_time):.3f} seconds")
        print("-" * 50)

if __name__ == "__main__":
    test_fast_rag() 