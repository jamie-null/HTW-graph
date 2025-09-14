#!/usr/bin/env python3
"""
Test script to check llama.cpp embedding server response format
"""

import json
import requests

def test_embedding_response():
    """Test the embedding endpoint and show exact response format."""
    url = "http://localhost:8080/embedding"
    test_text = "test embedding response"
    
    print(f"[TEST] Sending request to: {url}")
    print(f"[TEST] Test text: '{test_text}'")
    
    try:
        response = requests.post(
            url,
            json={"content": test_text},
            timeout=10
        )
        
        print(f"[RESPONSE] Status code: {response.status_code}")
        print(f"[RESPONSE] Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[RESPONSE] Type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"[RESPONSE] Keys: {list(result.keys())}")
                for key, value in result.items():
                    print(f"[RESPONSE] '{key}': {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            elif isinstance(result, list):
                print(f"[RESPONSE] List length: {len(result)}")
                print(f"[RESPONSE] First few items: {result[:5] if len(result) > 5 else result}")
                if result and isinstance(result[0], (int, float)):
                    print(f"[RESPONSE] This looks like a direct embedding vector!")
            
            print(f"[RESPONSE] Full response (truncated): {str(result)[:500]}...")
            
        else:
            print(f"[ERROR] Response: {response.text}")
            
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")

if __name__ == "__main__":
    test_embedding_response()
