#!/usr/bin/env python3

from ragen.env.sokoban import SokobanEnv

def test_action_parsing():
    """Test the improved action parsing functionality."""
    
    env = SokobanEnv(dual_agent=True)
    
    # Test cases from the log - problematic model outputs
    test_cases = [
        # Good cases
        ("<answer>Up</answer>", 1),
        ("<answer>Down</answer>", 2), 
        ("<answer>Left</answer>", 3),
        ("<answer>Right</answer>", 4),
        ("<answer>1</answer>", 1),
        ("<answer>2</answer>", 2),
        ("<answer>3</answer>", 3),
        ("<answer>4</answer>", 4),
        
        # Real model output cases
        ("Right", 4),
        ("Up", 1),
        ("Move: Up, Down, Left, Right", 1),  # Should extract "Up"
        ("Human intelligence is a complex and tricky thing...", 0),  # Should be invalid
        ("I think we should move Up to push the box", 1),  # Should extract "Up"
        ("The best action is 2", 2),  # Should extract number
        
        # Edge cases
        ("", 0),
        (None, 0),
        ("invalid text", 0),
        ("move left now", 3),
        ("go right please", 4),
    ]
    
    print("Testing action parsing...")
    print("=" * 50)
    
    for i, (input_text, expected) in enumerate(test_cases):
        result = env.extract_action(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Test {i+1:2d}: {result} (expected {expected})")
        print(f"   Input: {repr(input_text)}")
        if result != expected:
            print(f"   ERROR: Got {result}, expected {expected}")
        print()
    
    # Test summary
    passed = sum(1 for input_text, expected in test_cases 
                if env.extract_action(input_text) == expected)
    total = len(test_cases)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    test_action_parsing() 