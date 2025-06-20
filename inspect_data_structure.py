#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re

def inspect_data():
    """Inspect the detailed data structure."""
    
    df = pd.read_parquet('data/sokoban/train_dual_agent.parquet')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check first example
    example = df.iloc[0]
    print(f"\nFirst example keys: {list(example.keys())}")
    
    prompt = example['prompt']
    print(f"Prompt type: {type(prompt)}")
    
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
        
    if isinstance(prompt, list) and len(prompt) > 0:
        print(f"Prompt length: {len(prompt)}")
        print(f"First element type: {type(prompt[0])}")
        
        # Show first few elements
        for i, elem in enumerate(prompt[:3]):
            print(f"\nElement {i}:")
            if isinstance(elem, dict):
                print(f"  Keys: {list(elem.keys())}")
                if 'role' in elem:
                    print(f"  Role: {elem['role']}")
                if 'content' in elem:
                    content = elem['content']
                    print(f"  Content length: {len(content)}")
                    print(f"  Content preview: {content[:200]}...")
                    
                    # Check for answer tags
                    if '<answer>' in content.lower():
                        print(f"  ✓ Contains <answer> tags")
                    else:
                        print(f"  ✗ No <answer> tags found")
            else:
                print(f"  Content: {str(elem)[:200]}...")

if __name__ == "__main__":
    inspect_data() 