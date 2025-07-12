import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Parameters
np.random.seed(42)  # For reproducibility
grid_size = 10  # 10x10 grid
horizon = 1000  # Number of time steps
gamma = 0.1  # Exploration parameter for EXP3
erasure_prob = 0.1  # Probability an action is erased
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
action_names = ['Up', 'Down', 'Left', 'Right']
start_state = (0, 0)
goal_state = (9, 9)

# Reward function: Negative Manhattan distance to goal, 0 at goal
def get_reward(state, action):
    if state == goal_state:
        return 0  # No reward at goal
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
        next_x, next_y = state
    return - (abs(9 - next_x) + abs(9 - next_y))

# EXP3 algorithm with action erasure
def exp3_grid_with_erasure(horizon, grid_size, gamma, erasure_prob):
    weights = np.ones((grid_size, grid_size, 4))  # Weights for each state-action pair
    rewards = np.zeros(horizon)
    cumulative_regret = np.zeros(horizon)
    path = [start_state]
    current_state = start_state
    optimal_reward_per_step = -1  # Simplified optimal reward per step
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
            # If all weights are zero (shouldn't happen), uniform over available
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

# Create reward grid for visualization
def get_reward_grid():
    reward_grid = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) == goal_state:
                reward_grid[x, y] = 0
            else:
                rewards = [get_reward((x, y), action) for action in actions]
                reward_grid[x, y] = max(rewards)
    return reward_grid

# Run EXP3 with erasure
path, rewards, cumulative_regret, reached_goal = exp3_grid_with_erasure(horizon, grid_size, gamma, erasure_prob)

# Set up figure for animation
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Grid setup
ax1.set_xlim(-0.5, grid_size-0.5)
ax1.set_ylim(-0.5, grid_size-0.5)
ax1.set_xticks(range(grid_size))
ax1.set_yticks(range(grid_size))
ax1.grid(True)
ax1.set_title('Learner Path on 10x10 Grid with Erasure')
ax1.plot(0, 0, 'go', label='Start (0,0)')
ax1.plot(9, 9, 'r*', label='Goal (9,9)')
path_line, = ax1.plot([], [], 'b.-', label='Path')
ax1.legend()

# Regret plot setup
regret_line, = ax2.plot([], [], label='Cumulative Regret')
ax2.set_xlim(0, horizon)
ax2.set_ylim(0, max(cumulative_regret) * 1.1)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Cumulative Regret')
ax2.set_title('Cumulative Regret vs Horizon')
ax2.legend()

# Animation function
def update(frame):
    if frame < len(path):
        x, y = zip(*path[:frame+1])
        path_line.set_data(x, y)
        regret_line.set_data(range(frame+1), cumulative_regret[:frame+1])
    return path_line, regret_line

# Create and run animation
ani = FuncAnimation(fig, update, frames=horizon, interval=50, blit=True)
plt.tight_layout()
plt.show()

# Reward grid visualization
reward_grid = get_reward_grid()
fig2, ax3 = plt.subplots(figsize=(6, 5))
sns.heatmap(reward_grid, annot=True, fmt='.1f', cmap='viridis', ax=ax3)
ax3.set_title('Reward Grid (Max Reward per State)')
ax3.set_xlabel('Y Coordinate')
ax3.set_ylabel('X Coordinate')
plt.show()

# Print results
print(f"Total reward: {np.sum(rewards):.2f}")
print(f"Final cumulative regret: {cumulative_regret[-1]:.2f}")
print(f"Final state: {path[-1]}")
print(f"Reached goal (9,9): {reached_goal}")
