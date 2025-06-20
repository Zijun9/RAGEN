#!/usr/bin/env python3

from ragen.env.sokoban import SokobanEnv
import numpy as np

def test_single_agent_invalid_actions():
    print("=== 单智能体模式无效动作测试 ===")
    env = SokobanEnv(dual_agent=False, dim_room=(6,6), num_boxes=1)
    obs = env.reset()
    
    print("初始状态:")
    print(obs)
    print()
    
    # 测试各种动作
    actions = [1, 2, 3, 4]  # Up, Down, Left, Right
    action_names = ["Up", "Down", "Left", "Right"]
    
    for action, name in zip(actions, action_names):
        env_copy = env.copy()
        obs, reward, done, info = env_copy.step(action)
        print(f"{name}动作: reward={reward:.1f}, effective={info.get('action_is_effective', 'N/A')}")
    
    print()

def test_dual_agent_invalid_actions():
    print("=== 双智能体模式无效动作测试 ===")
    env = SokobanEnv(dual_agent=True, dim_room=(6,6), num_boxes=1)
    obs = env.reset()
    
    print("初始状态:")
    print(obs)
    print()
    
    # 测试智能体P的动作限制
    print("智能体P（只能上下移动）:")
    p_actions = [1, 2, 3, 4]  # Up, Down, Left, Right
    p_action_names = ["Up", "Down", "Left", "Right"]
    
    for action_p, name in zip(p_actions, p_action_names):
        env_copy = env.copy()
        # P做动作，Q不动（无效动作0）
        obs, reward, done, info = env_copy.step((action_p, 0))
        print(f"  P-{name}: reward={reward:.1f}, P_invalid={info.get('agent1_invalid_action', 'N/A')}")
    
    print()
    print("智能体Q（只能左右移动）:")
    q_actions = [1, 2, 3, 4]  # Up, Down, Left, Right  
    q_action_names = ["Up", "Down", "Left", "Right"]
    
    for action_q, name in zip(q_actions, q_action_names):
        env_copy = env.copy()
        # Q做动作，P不动（无效动作0）
        obs, reward, done, info = env_copy.step((0, action_q))
        print(f"  Q-{name}: reward={reward:.1f}, Q_invalid={info.get('agent2_invalid_action', 'N/A')}")

def test_wall_collision():
    print("\n=== 撞墙测试 ===")
    
    # 单智能体撞墙
    print("单智能体撞墙:")
    env = SokobanEnv(dual_agent=False, dim_room=(6,6), num_boxes=1)
    obs = env.reset()
    
    # 连续向上移动直到撞墙
    for i in range(10):
        prev_obs = obs
        obs, reward, done, info = env.step(1)  # Up
        effective = info.get('action_is_effective', True)
        print(f"  第{i+1}次Up: reward={reward:.1f}, effective={effective}")
        
        if not effective:
            print("  -> 撞墙了！")
            break
        if obs == prev_obs:
            print("  -> 状态没变化，可能撞墙了")
            break
    
    print()
    
    # 双智能体撞墙
    print("双智能体撞墙:")
    env = SokobanEnv(dual_agent=True, dim_room=(6,6), num_boxes=1)
    obs = env.reset()
    
    # P向上移动直到撞墙，Q保持不动
    for i in range(10):
        prev_obs = obs
        obs, reward, done, info = env.step((1, 3))  # P-Up, Q-Left
        p_effective = info.get('agent1_effective', True)
        q_effective = info.get('agent2_effective', True)
        print(f"  第{i+1}次(P-Up,Q-Left): reward={reward:.1f}, P_eff={p_effective}, Q_eff={q_effective}")
        
        if not p_effective and not q_effective:
            print("  -> 两个智能体都无法移动了！")
            break
        if obs == prev_obs:
            print("  -> 状态没变化")
            break

if __name__ == "__main__":
    test_single_agent_invalid_actions()
    test_dual_agent_invalid_actions() 
    test_wall_collision() 