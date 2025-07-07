import pygame
import random
import numpy as np

# -----------------------------------
# 🎲 Bandit Parameters
# -----------------------------------
num_arms = 4
horizon = 1000
explore_rounds_per_arm = 100  # 10 rounds per arm = 40 steps
explore_steps = explore_rounds_per_arm * num_arms
reward_probs = [0.2, 0.5, 0.35, 0.8]  # true reward probabilities per arm

# -----------------------------------
# 🧠 ETC Algorithm State
# -----------------------------------
counts = [0] * num_arms        # number of times each arm is pulled
values = [0.0] * num_arms      # empirical mean of each arm
cumulative_reward = 0
step = 0
history = []  # (step, chosen_arm, reward)

# -----------------------------------
# 🎨 Pygame Setup
# -----------------------------------
pygame.init()
width, height = 800, 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("🎰 ETC Multi-Armed Bandit Simulation")
font = pygame.font.SysFont("monospace", 20)
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (70, 130, 180)
GRAY = (169, 169, 169)
ARM_COLORS = [(255, 99, 71), (30, 144, 255), (60, 179, 113), (255, 215, 0)]

# -----------------------------------
# 🕹️ Functions
# -----------------------------------
def pull_arm(arm):
    """Simulate pulling an arm: Bernoulli reward"""
    return 1 if random.random() < reward_probs[arm] else 0

def draw_screen():
    screen.fill(WHITE)
    
    spacing = width // (num_arms + 1)
    for i in range(num_arms):
        x = (i + 1) * spacing
        height_bar = int(values[i] * 300)
        pygame.draw.rect(screen, ARM_COLORS[i], (x - 25, height - height_bar - 100, 50, height_bar))
        
        # Arm label
        label = font.render(f"Arm {i+1}", True, BLACK)
        screen.blit(label, (x - 30, height - 80))
        
        # Avg reward display
        avg = font.render(f"{values[i]:.2f}", True, BLACK)
        screen.blit(avg, (x - 20, height - height_bar - 130))
        
        # Times pulled
        pulls = font.render(f"{counts[i]} pulls", True, GRAY)
        screen.blit(pulls, (x - 40, height - 50))

    # Display phase
    if step < explore_steps:
        phase = "Exploration 🧪"
    elif step < horizon:
        phase = "Exploitation 💥"
    else:
        phase = "Horizon Reached 🚧"

    phase_text = font.render(f"Phase: {phase}", True, BLUE)
    screen.blit(phase_text, (10, 10))
    
    # Step and reward info
    step_text = font.render(f"Step: {step}", True, BLACK)
    reward_text = font.render(f"Cumulative Reward: {cumulative_reward}", True, BLACK)
    screen.blit(step_text, (10, 40))
    screen.blit(reward_text, (10, 70))

    # Last action
    if history:
        last = history[-1]
        last_text = font.render(f"Chose Arm {last[1]+1} → Reward: {last[2]}", True, BLACK)
        screen.blit(last_text, (10, 100))

    pygame.display.flip()

# -----------------------------------
# 🎯 ETC Core Logic
# -----------------------------------
def choose_arm(step):
    if step < explore_steps:
        return step % num_arms  # Round robin
    else:
        return int(np.argmax(values))  # Commit to best arm

# -----------------------------------
# 🧪 Main Loop
# -----------------------------------
running = True
paused = False

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Optional: spacebar to pause/unpause
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if step < horizon and not paused:
        arm = choose_arm(step)
        reward = pull_arm(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        cumulative_reward += reward
        history.append((step + 1, arm, reward))
        step += 1

    draw_screen()
    clock.tick(10)  # Control simulation speed (FPS)

pygame.quit()
