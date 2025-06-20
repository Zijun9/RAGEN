#!/usr/bin/env python3

from ragen.env.sokoban import SokobanEnv

def debug_actual_predictions():
    """Debug the actual predictions from training log."""
    
    env = SokobanEnv(dual_agent=True)
    
    # Real predictions from the training log
    actual_predictions = [
        'Move Up',
        'Move Down', 
        'Move Left',
        'Move Right',
        "Human intelligence is a complex and tricky thing...",
        "Human: I'm stuck on this puzzle. Can you help me solve it?..."
    ]
    
    print("Testing ACTUAL predictions from training log:")
    print("=" * 60)
    
    for i, prediction in enumerate(actual_predictions):
        result = env.extract_action(prediction)
        print(f"Test {i+1}:")
        print(f"  Input: {repr(prediction[:50])}{'...' if len(prediction) > 50 else ''}")
        print(f"  Result: {result}")
        print(f"  Expected: {'1 or 2' if 'Up' in prediction or 'Down' in prediction else '3 or 4' if 'Left' in prediction or 'Right' in prediction else '0'}")
        print()

if __name__ == "__main__":
    debug_actual_predictions() 