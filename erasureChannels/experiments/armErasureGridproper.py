import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Parameters
np.random.seed(42)  # For reproducibility
grid_size = 10  # 10x10 grid
horizon = 10000  # Number of time steps
gamma = 0.1  # Exploration parameter for EXP3
erasure_prob = 0.3 # Probability an action is erased
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
start_state = (0, 0)
goal_state = (9, 9)
n_runs = 100  # Number of runs to average over

# Reward function: Normalized negative Manhattan distance to goal, 0 at goal
def get_reward(state, action):
    if state == goal_state:
        return 0  # No reward at goal
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
        next_x, next_y = state
    manhattan_dist = abs(9 - next_x) + abs(9 - next_y)
    return -manhattan_dist / 18.0  # Normalize by dividing by 18

# EXP3 algorithm with action erasure
def exp3_grid_with_erasure(horizon, grid_size, gamma, erasure_prob):
    weights = np.ones((grid_size, grid_size, 4))  # Weights for each state-action pair
    rewards = np.zeros(horizon)
    cumulative_regret = np.zeros(horizon)
    path = [start_state]
    current_state = start_state
    optimal_reward_per_step = -1 / 18.0  # Normalized optimal reward per step
    reached_goal = False

    for t in range(horizon):
        x, y = current_state
        if current_state == goal_state:
            reached_goal = True
            rewards[t] = 0
            path.append(current_state)
            cumulative_regret[t] = cumulative_regret[t-1] if t > 0 else 0
            continue
        
        # Determine available actions (1 = available, 0 = erased)
        available_actions = np.random.binomial(1, 1 - erasure_prob, 4)
        if np.sum(available_actions) == 0:
            # If all actions are erased, stay in place
            reward = get_reward(current_state, (0, 0))  # No movement
            rewards[t] = reward
            path.append(current_state)
            regret = optimal_reward_per_step - reward
            cumulative_regret[t] = cumulative_regret[t-1] + regret if t > 0 else regret
            continue
        
        # Get weights for available actions
        w = weights[x, y] * available_actions  # Zero weight for erased actions
        w_sum = np.sum(w)
        if w_sum == 0:
            # If all weights are zero, uniform over available actions
            probs = available_actions / np.sum(available_actions)
        else:
            probs = (1 - gamma) * (w / w_sum) + gamma * available_actions / np.sum(available_actions)
        
        # Choose action from available actions
        chosen_action_idx = np.random.choice(4, p=probs)
        chosen_action = actions[chosen_action_idx]
        
        # Get reward
        reward = get_reward(current_state, chosen_action)
        rewards[t] = reward
        
        # Update next state
        next_x = x + chosen_action[0]
        next_y = y + chosen_action[1]
        if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
            current_state = (next_x, next_y)
        path.append(current_state)
        
        # Update weights for chosen action
        estimated_reward = reward / probs[chosen_action_idx]
        weights[x, y, chosen_action_idx] *= np.exp(gamma * estimated_reward / 4)
        
        # Calculate regret
        regret = optimal_reward_per_step - reward
        cumulative_regret[t] = cumulative_regret[t-1] + regret if t > 0 else regret
    
    return path, rewards, cumulative_regret, reached_goal

# Run multiple simulations and average results
def run_multiple_simulations(n_runs, horizon, grid_size, gamma, erasure_prob):
    all_cumulative_regrets = np.zeros((n_runs, horizon))
    goal_reached_count = 0

    for run in tqdm(range(n_runs)):
        path, rewards, cumulative_regret, reached_goal = exp3_grid_with_erasure(horizon, grid_size, gamma, erasure_prob)
        all_cumulative_regrets[run] = cumulative_regret
        if reached_goal:
            goal_reached_count += 1

    # Average cumulative regret across runs
    avg_cumulative_regret = np.mean(all_cumulative_regrets, axis=0)
    goal_reached_rate = goal_reached_count / n_runs

    return avg_cumulative_regret, goal_reached_rate

# Run simulations
avg_cumulative_regret, goal_reached_rate = run_multiple_simulations(n_runs, horizon, grid_size, gamma, erasure_prob)

# Plot averaged cumulative regret
plt.figure(figsize=(10, 6))
plt.plot(avg_cumulative_regret, label='Averaged Cumulative Regret')
plt.xlabel('Time Step')
plt.ylabel('Averaged Cumulative Regret')
plt.title(f'Averaged Cumulative Regret over {n_runs} Runs (Horizon: {horizon})')
plt.legend()
plt.show()

# Print results
print(f"Average final cumulative regret: {avg_cumulative_regret[-1]:.2f}")
print(f"Goal reached rate: {goal_reached_rate:.2%}")
