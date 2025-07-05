"""
Simple utility to load documents from JSON file for RAG system.
"""

import os
import json
from typing import List

def load_documents_from_json(file_path: str) -> List[str]:
    """Load documents from a JSON file."""
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # If JSON has categories, flatten them
                for category, items in data.items():
                    if isinstance(items, list):
                        documents.extend([str(item) for item in items])
            elif isinstance(data, list):
                documents = [str(item) for item in data]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {file_path}")
    return documents

def load_company_documents() -> List[str]:
    """Load company documents from JSON file."""
    json_file = "data/company_documents.json"
    return load_documents_from_json(json_file) 