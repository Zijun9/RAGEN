#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re

def show_answer_examples():
    """Show specific examples with answer tags."""
    
    df = pd.read_parquet('data/sokoban/train_dual_agent.parquet')
    example = df.iloc[0]
    prompt = example['prompt']
    
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    
    for i, msg in enumerate(prompt):
        print(f"\n=== Message {i+1} (Role: {msg['role']}) ===")
        content = msg['content']
        
        # Find and extract answer tags
        answer_matches = re.findall(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
        
        if answer_matches:
            print(f"Found {len(answer_matches)} answer tags:")
            for j, answer in enumerate(answer_matches):
                print(f"  Answer {j+1}: '{answer.strip()}'")
        
        # Show content around answer tags
        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            if '<answer>' in line.lower():
                print(f"\nContext around answer (line {line_num+1}):")
                start = max(0, line_num - 2)
                end = min(len(lines), line_num + 3)
                for k in range(start, end):
                    marker = ">>> " if k == line_num else "    "
                    print(f"{marker}{lines[k]}")

if __name__ == "__main__":
    show_answer_examples() 