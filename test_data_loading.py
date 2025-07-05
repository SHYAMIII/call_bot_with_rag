#!/usr/bin/env python3
"""
Test script to demonstrate JSON data loading for RAG
"""

from data.load_documents import load_company_documents

def test_data_loading():
    """Test JSON data loading."""
    print("Testing JSON data loading...")
    
    # Load documents from JSON file
    documents = load_company_documents()
    print(f"Loaded {len(documents)} documents from JSON file")
    
    # Show some examples
    print("\nSample documents:")
    for i, doc in enumerate(documents[:5]):
        print(f"  {i+1}. {doc[:80]}...")
    
    print(f"\nTotal documents available: {len(documents)}")
    
    # Test categories
    print("\nDocument categories found:")
    categories = set()
    for doc in documents:
        if "Salesforce" in doc:
            categories.add("Salesforce")
        if "AI" in doc or "machine learning" in doc:
            categories.add("AI/ML")
        if "contact" in doc.lower() or "phone" in doc.lower():
            categories.add("Contact")
    
    for category in categories:
        print(f"  - {category}")

if __name__ == "__main__":
    test_data_loading() 