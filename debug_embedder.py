#!/usr/bin/env python3
"""
Debug embedder - simplified version to isolate the parsing issue
"""

import json
import requests
from pathlib import Path

def debug_single_embedding():
    """Test single embedding and save debug info."""
    
    # Test the server
    print("[DEBUG] Testing embedding server...")
    
    try:
        response = requests.post(
            "http://localhost:8080/embedding",
            json={"content": "test skill: Python (5 years, Advanced)"},
            timeout=30
        )
        
        print(f"[DEBUG] Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Save full response to file for inspection
            with open('debug_response.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[DEBUG] Full response saved to debug_response.json")
            
            print(f"[DEBUG] Response type: {type(result)}")
            
            if isinstance(result, list):
                print(f"[DEBUG] List length: {len(result)}")
                if len(result) > 0:
                    print(f"[DEBUG] First item type: {type(result[0])}")
                    if isinstance(result[0], dict):
                        print(f"[DEBUG] First item keys: {list(result[0].keys())}")
                        if 'embedding' in result[0]:
                            embedding = result[0]['embedding']
                            print(f"[DEBUG] Embedding type: {type(embedding)}")
                            print(f"[DEBUG] Embedding length: {len(embedding) if isinstance(embedding, list) else 'not list'}")
                            print(f"[DEBUG] First 5 values: {embedding[:5] if isinstance(embedding, list) else 'not list'}")
                            
                            # Handle nested embedding format: [[actual_embedding_values]]
                            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                                print("[DEBUG] Found nested embedding format, extracting inner list")
                                embedding = embedding[0]
                                print(f"[DEBUG] Inner embedding length: {len(embedding)}")
                                print(f"[DEBUG] Inner first 5 values: {embedding[:5]}")
                            
                            # Test if this is what we need
                            if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                                print("[SUCCESS] Found valid embedding!")
                                return embedding
                            else:
                                print("[ERROR] Embedding is not a valid numeric list")
                        else:
                            print("[ERROR] No 'embedding' key found")
                    else:
                        print(f"[ERROR] First item is not a dict: {result[0]}")
            elif isinstance(result, dict):
                print(f"[DEBUG] Dict keys: {list(result.keys())}")
            
        else:
            print(f"[ERROR] Server error: {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
    
    return None

if __name__ == "__main__":
    embedding = debug_single_embedding()
    if embedding:
        print(f"[SUCCESS] Got embedding with {len(embedding)} dimensions")
    else:
        print("[FAILED] Could not extract embedding")
