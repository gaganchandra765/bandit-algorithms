import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Parameters
np.random.seed(42)  # For reproducibility
grid_size = 10  # 10x10 grid
horizon = 10000  # Number of time steps
gamma = 0.1  # Exploration parameter for EXP3
erasure_probs = [0.1, 0.3, 0.5]  # Test multiple erasure probabilities
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
action_names = ['Up', 'Down', 'Left', 'Right']
start_state = (0, 0)
goal_state = (9, 9)
n_runs = 100  # Number of runs to average over

# Reward function: Normalized negative Manhattan distance to goal
def get_reward(state, action):
    if state == goal_state:
        return 0  # No reward at goal
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
        next_x, next_y = state
    manhattan_dist = abs(9 - next_x) + abs(9 - next_y)
    return -manhattan_dist / 18.0  # Normalize by dividing by 18

# Single EXP3 algorithm (baseline)
def exp3_grid_single(horizon, grid_size, gamma):
    weights = np.ones((grid_size, grid_size, 4))  # Agent's weights
    rewards = np.zeros(horizon)
    cumulative_regret = np.zeros(horizon)
    path = [start_state]
    current_state = start_state
    optimal_reward_per_step = -1 / 18.0  # Normalized optimal reward
    reached_goal = False

    for t in range(horizon):
        x, y = current_state
        if current_state == goal_state:
            reached_goal = True
            rewards[t] = 0
            path.append(current_state)
            cumulative_regret[t] = cumulative_regret[t-1] if t > 0 else 0
            continue

        w = weights[x, y]
        w_sum = np.sum(w)
        probs = (1 - gamma) * (w / w_sum) + gamma / 4
        chosen_action_idx = np.random.choice(4, p=probs)
        chosen_action = actions[chosen_action_idx]

        reward = get_reward(current_state, chosen_action)
        rewards[t] = reward

        next_x = x + chosen_action[0]
        next_y = y + chosen_action[1]
        if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
            current_state = (next_x, next_y)
        path.append(current_state)

        est_reward = reward / probs[chosen_action_idx]
        weights[x, y, chosen_action_idx] *= np.exp(gamma * est_reward / 4)

        regret = optimal_reward_per_step - reward
        cumulative_regret[t] = cumulative_regret[t-1] + regret if t > 0 else regret

    return path, rewards, cumulative_regret, reached_goal

# Dual EXP3 algorithm with learner and agent
def exp3_grid_dual(horizon, grid_size, gamma, erasure_prob):
    learner_weights = np.ones((grid_size, grid_size, 4))  # Learner's weights
    agent_weights = np.ones((grid_size, grid_size, 4))  # Agent's weights
    rewards = np.zeros(horizon)
    cumulative_regret = np.zeros(horizon)
    path = [start_state]
    current_state = start_state
    optimal_reward_per_step = -1 / 18.0  # Normalized optimal reward
    reached_goal = False

    for t in range(horizon):
        x, y = current_state
        if current_state == goal_state:
            reached_goal = True
            rewards[t] = 0
            path.append(current_state)
            cumulative_regret[t] = cumulative_regret[t-1] if t > 0 else 0
            continue

        # Learner's EXP3: Suggest an action
        learner_w = learner_weights[x, y]
        learner_w_sum = np.sum(learner_w)
        learner_probs = (1 - gamma) * (learner_w / learner_w_sum) + gamma / 4
        learner_action_idx = np.random.choice(4, p=learner_probs)

        # Agent's EXP3: Select an action
        agent_w = agent_weights[x, y]
        agent_w_sum = np.sum(agent_w)
        agent_probs = (1 - gamma) * (agent_w / agent_w_sum) + gamma / 4
        agent_action_idx = np.random.choice(4, p=agent_probs)

        # Signal transmission
        if np.random.rand() > erasure_prob:  # Signal received
            chosen_action_idx = learner_action_idx  # Use learner's action
        else:  # Signal erased, agent uses its own action
            chosen_action_idx = agent_action_idx

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

        # Update weights for both learner and agent
        learner_est_reward = reward / learner_probs[chosen_action_idx]
        learner_weights[x, y, chosen_action_idx] *= np.exp(gamma * learner_est_reward / 4)

        agent_est_reward = reward / agent_probs[chosen_action_idx]
        agent_weights[x, y, chosen_action_idx] *= np.exp(gamma * agent_est_reward / 4)

        # Calculate regret
        regret = optimal_reward_per_step - reward
        cumulative_regret[t] = cumulative_regret[t-1] + regret if t > 0 else regret

    return path, rewards, cumulative_regret, reached_goal

# Run multiple simulations and average results
def run_multiple_simulations(n_runs, horizon, grid_size, gamma, erasure_prob, dual=True):
    all_cumulative_regrets = np.zeros((n_runs, horizon))
    goal_reached_count = 0

    for run in tqdm(range(n_runs), desc=f"Running simulations (Erasure Prob: {erasure_prob if dual else 'Single'})"):
        if dual:
            path, rewards, cumulative_regret, reached_goal = exp3_grid_dual(horizon, grid_size, gamma, erasure_prob)
        else:
            path, rewards, cumulative_regret, reached_goal = exp3_grid_single(horizon, grid_size, gamma)
        all_cumulative_regrets[run] = cumulative_regret
        if reached_goal:
            goal_reached_count += 1

    avg_cumulative_regret = np.mean(all_cumulative_regrets, axis=0)
    goal_reached_rate = goal_reached_count / n_runs

    return avg_cumulative_regret, goal_reached_rate

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

# Run simulations for different erasure probabilities and single EXP3
plt.figure(figsize=(12, 6))
results = []
for erasure_prob in erasure_probs:
    avg_cumulative_regret, goal_reached_rate = run_multiple_simulations(n_runs, horizon, grid_size, gamma, erasure_prob, dual=True)
    results.append((avg_cumulative_regret, goal_reached_rate, f'Dual EXP3 (Erasure Prob: {erasure_prob})'))
    plt.plot(avg_cumulative_regret, label=f'Dual EXP3 (Erasure Prob: {erasure_prob})')

# Run single EXP3 baseline
avg_cumulative_regret, goal_reached_rate = run_multiple_simulations(n_runs, horizon, grid_size, gamma, 0.3, dual=False)
results.append((avg_cumulative_regret, goal_reached_rate, 'Single EXP3'))

# Plot settings
plt.xlabel('Time Step')
plt.ylabel('Averaged Cumulative Regret')
plt.title(f'Averaged Cumulative Regret over {n_runs} Runs (Horizon: {horizon})')
plt.legend()
plt.grid(True)
plt.show()

# Reward grid visualization
reward_grid = get_reward_grid()
plt.figure(figsize=(6, 5))
sns.heatmap(reward_grid, annot=True, fmt='.2f', cmap='viridis')
plt.title('Reward Grid (Max Reward per State)')
plt.xlabel('Y Coordinate')
plt.ylabel('X Coordinate')
plt.show()

# Print results
for avg_cumulative_regret, goal_reached_rate, label in results:
    print(f"\n{label}:")
    print(f"Average final cumulative regret: {avg_cumulative_regret[-1]:.2f}")
    print(f"Goal reached rate: {goal_reached_rate:.2%}")