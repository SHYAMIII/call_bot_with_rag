#!/usr/bin/env python3
"""
Debug script to test RAG with contact questions
"""

from llm.fast_rag import initialize_fast_rag, get_fast_rag_response
import time

def test_contact_questions():
    """Test RAG with contact-related questions."""
    print("Testing RAG with contact questions...")
    
    # Initialize RAG
    rag = initialize_fast_rag()
    
    # Test questions about contact
    test_questions = [
        "What is the contact information for Technology Mindz?",
        "How can I contact Technology Mindz?",
        "What is the phone number for Technology Mindz?",
        "What is the email for Technology Mindz?",
        "How do I reach Technology Mindz?",
        "Contact details of Technology Mindz",
        "Phone number and email for Technology Mindz"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        answer, timings = rag.query(question, return_timings=True)
        end_time = time.time()
        
        print(f"Answer: {answer}")
        print(f"Retrieval time: {timings['retrieval_time']:.3f}s")
        print(f"Generation time: {timings['generation_time']:.3f}s")
        print(f"Total time: {timings['total_time']:.3f}s")
        
        # Test retrieval separately
        print(f"\n--- Retrieval Debug ---")
        retrieved_docs = rag.retrieve(question, top_k=3)
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc[:100]}...")

if __name__ == "__main__":
    test_contact_questions() 