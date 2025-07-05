#!/usr/bin/env python3
"""
Performance test script for optimized RAG system
"""

from llm.fast_rag import initialize_fast_rag, get_fast_rag_response
import time
import statistics

def test_performance():
    """Test the performance of the optimized RAG system."""
    print("Testing optimized RAG performance...")
    
    # Initialize RAG
    rag = initialize_fast_rag()
    
    # Test questions
    test_questions = [
        "What is the contact information for Technology Mindz?",
        "What services do you offer?",
        "What is your phone number?",
        "How can I reach you?",
        "What AI services do you provide?",
        "Tell me about your Salesforce services",
        "What is your email address?",
        "How do I contact Technology Mindz?",
        "What are your technology expertise?",
        "What is your phone number and email?"
    ]
    
    # Performance metrics
    retrieval_times = []
    generation_times = []
    total_times = []
    cache_hits = 0
    
    print(f"\nRunning {len(test_questions)} queries...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuery {i}: {question}")
        
        # First query (cold)
        start_time = time.time()
        answer, timings = rag.query(question, return_timings=True)
        first_query_time = time.time() - start_time
        
        # Second query (warm cache)
        start_time = time.time()
        answer2, timings2 = rag.query(question, return_timings=True)
        second_query_time = time.time() - start_time
        
        # Check if cached
        if second_query_time < first_query_time * 0.5:  # If significantly faster
            cache_hits += 1
            cache_status = "[CACHED]"
        else:
            cache_status = "[FRESH]"
        
        print(f"  First query:  {first_query_time:.3f}s")
        print(f"  Second query: {second_query_time:.3f}s {cache_status}")
        print(f"  Retrieval:    {timings['retrieval_time']:.3f}s")
        print(f"  Generation:   {timings['generation_time']:.3f}s")
        print(f"  Answer:       {answer[:100]}...")
        
        retrieval_times.append(timings['retrieval_time'])
        generation_times.append(timings['generation_time'])
        total_times.append(timings['total_time'])
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Cache hit rate: {cache_hits}/{len(test_questions)} ({cache_hits/len(test_questions)*100:.1f}%)")
    print(f"Average retrieval time: {statistics.mean(retrieval_times):.3f}s")
    print(f"Average generation time: {statistics.mean(generation_times):.3f}s")
    print(f"Average total time: {statistics.mean(total_times):.3f}s")
    print(f"Min total time: {min(total_times):.3f}s")
    print(f"Max total time: {max(total_times):.3f}s")
    print(f"95th percentile: {statistics.quantiles(total_times, n=20)[18]:.3f}s")

if __name__ == "__main__":
    test_performance() 