#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re

def check_answer_format():
    """Check if the dataset uses <answer> tags correctly."""
    
    df = pd.read_parquet('data/sokoban/train_dual_agent.parquet')
    print(f"Dataset shape: {df.shape}")
    
    # Check a few examples
    for i in range(min(3, len(df))):
        example = df.iloc[i]
        prompt = example['prompt']
        
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        
        print(f"\n=== Example {i+1} ===")
        if isinstance(prompt, list):
            # Find assistant responses with answers
            for j, msg in enumerate(prompt):
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    content = msg['content']
                    # Check if it contains <answer> tags
                    answer_matches = re.findall(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
                    if answer_matches:
                        print(f"  Message {j}: Found {len(answer_matches)} answer tags")
                        for k, answer in enumerate(answer_matches):
                            print(f"    Answer {k+1}: {answer.strip()}")
                    else:
                        print(f"  Message {j}: No answer tags found")
                        print(f"    Content preview: {content[:100]}...")

if __name__ == "__main__":
    check_answer_format() 