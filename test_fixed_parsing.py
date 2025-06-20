#!/usr/bin/env python3

from ragen.env.sokoban import SokobanEnv

def test_fixed_postprocess():
    """Test the fixed postprocess_predictions method."""
    
    # Create dual agent environment
    env = SokobanEnv(dual_agent=True)
    envs = [env]
    
    # Test predictions from the actual training log
    test_predictions = [
        'Move Up',
        'Move Down', 
        'Move Left',
        'Move Right',
        "Human intelligence is a complex and tricky thing...",
        '<answer>Up</answer>',  # Standard format
        '<answer>2</answer>',   # Number format
    ]
    
    print("Testing FIXED postprocess_predictions method:")
    print("=" * 60)
    
    for i, prediction in enumerate(test_predictions):
        actions, action_is_valid = SokobanEnv.postprocess_predictions(envs, [prediction])
        action = actions[0]
        valid = action_is_valid[0]
        
        print(f"Test {i+1}:")
        print(f"  Input: {repr(prediction[:50])}{'...' if len(prediction) > 50 else ''}")
        print(f"  Action: {action}")
        print(f"  Valid: {valid}")
        print(f"  Expected: Valid for first 6, invalid for 5th")
        print()

if __name__ == "__main__":
    test_fixed_postprocess() 