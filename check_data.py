import pandas as pd
import json
import numpy as np

# Load the dual agent dataset
df = pd.read_parquet('data/sokoban/train_dual_agent.parquet')

print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())

# Check the first example
example = df.iloc[0]
print('\nFirst example:')
print('Data source:', example['data_source'])
print('Prompt type:', type(example['prompt']))

# Print the prompt content
prompt = example['prompt']
if isinstance(prompt, np.ndarray):
    prompt = prompt.tolist()

if isinstance(prompt, list):
    print(f'Number of prompts: {len(prompt)}')
    for i, p in enumerate(prompt):
        print(f'\nPrompt {i+1}:')
        if isinstance(p, dict) and 'role' in p and 'content' in p:
            print(f'Role: {p["role"]}')
            content = p['content']
            # Print first 500 characters to see the structure
            print('Content:')
            print(content[:500] + '...' if len(content) > 500 else content)
        else:
            print('Content:', str(p)[:500] + '...' if len(str(p)) > 500 else str(p))
else:
    print('Prompt content:')
    print(str(prompt)[:500] + '...' if len(str(prompt)) > 500 else str(prompt))

print('\nAbility:', example['ability'])
print('Reward model:', example['reward_model'])
print('Extra info:', example['extra_info']) 