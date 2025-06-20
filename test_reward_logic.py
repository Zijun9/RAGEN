#!/usr/bin/env python3

from ragen.env.sokoban import SokobanEnv
import numpy as np

def test_reward_logic():
    print("=== Dual Agent Reward Logic Test ===")
    env = SokobanEnv(dual_agent=True, dim_room=(6,6), num_boxes=1)
    obs = env.reset()
    
    print("Initial state:")
    print(obs)
    print()
    
    print("🎯 Reward Rules:")
    print("  - Role constraint violation (P moves left/right, Q moves up/down): -1.0")
    print("  - Physical environment constraints (hitting walls, boundaries, etc.): -0.1") 
    print("  - Normal movement: -0.1")
    print("  - Push box to target: +1.0")
    print("  - Complete task: +10.0")
    print()
    
    test_cases = [
        # (action_p, action_q, description)
        (1, 3, "P-Up(valid), Q-Left(valid)"),
        (1, 1, "P-Up(valid), Q-Up(role violation)"),
        (3, 3, "P-Left(role violation), Q-Left(valid)"),
        (3, 1, "P-Left(role violation), Q-Up(role violation)"),
        (0, 0, "P-None(role violation), Q-None(role violation)"),
    ]
    
    for action_p, action_q, desc in test_cases:
        env_copy = env.copy()
        obs, reward, done, info = env_copy.step((action_p, action_q))
        
        p_invalid = info.get('agent1_invalid_action', False)
        q_invalid = info.get('agent2_invalid_action', False)
        p_effective = info.get('agent1_effective', False)
        q_effective = info.get('agent2_effective', False)
        
        print(f"📊 {desc}:")
        print(f"   Total reward: {reward:.1f}")
        print(f"   P violation: {p_invalid}, P effective: {p_effective}")
        print(f"   Q violation: {q_invalid}, Q effective: {q_effective}")
        
        # Analyze reward composition
        expected_reward = 0
        if p_invalid:
            expected_reward -= 1.0  # P role violation
        else:
            expected_reward -= 0.1  # P normal action (might be ineffective due to physical constraints, but still -0.1)
            
        if q_invalid:
            expected_reward -= 1.0  # Q role violation
        else:
            expected_reward -= 0.1  # Q normal action
            
        print(f"   Expected reward: {expected_reward:.1f}")
        print(f"   ✅ Correct" if abs(reward - expected_reward) < 0.01 else f"   ❌ Incorrect")
        print()

def test_physical_constraints():
    print("=== Physical Environment Constraints Test ===")
    
    # Create a specific layout to test wall collisions
    env = SokobanEnv(dual_agent=True, dim_room=(6,6), num_boxes=1)
    
    # Reset multiple times until finding a suitable layout
    for _ in range(10):
        obs = env.reset()
        # Check if P agent is near edge position for convenient wall collision testing
        if env.player_position[0] <= 1:  # P near top edge
            break
    
    print("Test layout:")
    print(obs)
    print()
    
    print("🧱 Physical constraint test:")
    
    # Test P moving up into wall
    env_copy = env.copy()
    obs, reward, done, info = env_copy.step((1, 3))  # P-Up, Q-Left
    
    p_effective = info.get('agent1_effective', False)
    q_effective = info.get('agent2_effective', False)
    
    print(f"P moves up (might hit wall):")
    print(f"  Reward: {reward:.1f}")
    print(f"  P effective: {p_effective}, Q effective: {q_effective}")
    
    if not p_effective:
        print(f"  ✅ P hit wall, but only deducted -0.1 (physical constraint)")
    else:
        print(f"  ℹ️  P moved successfully")
    print()

if __name__ == "__main__":
    test_reward_logic()
    test_physical_constraints() 