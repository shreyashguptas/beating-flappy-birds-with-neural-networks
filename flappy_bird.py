import pygame
import random
import numpy as np
from dqn_agent import DQNAgent
from config import Config
import torch
import csv
from datetime import datetime, timedelta

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 400, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont('comicsans', 30)

# Load images
BIRD_IMG = pygame.image.load('assets/bird.png')
BIRD_IMG = pygame.transform.scale(BIRD_IMG, (50, 50))
PIPE_IMG = pygame.image.load('assets/pipe.png')
BG_IMG = pygame.image.load('assets/background.png')
BG_IMG = pygame.transform.scale(BG_IMG, (WIDTH, HEIGHT))

# Player settings
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 50
PLAYER_X, PLAYER_Y = 100, HEIGHT // 2
GRAVITY = 0.5
JUMP_STRENGTH = -10

# Obstacle settings
OBSTACLE_WIDTH = 70
OBSTACLE_GAP = 200

# DQN Agent Initialization
config = Config()
agent = DQNAgent(config)

# Add these global variables
start_time = None
csv_file = 'game_logs.csv'

# Game Functions
def draw_player(x, y):
    SCREEN.blit(BIRD_IMG, (x, y))

def draw_obstacle(obstacles):
    for obs in obstacles:
        # Draw top pipe
        pipe_top = pygame.transform.flip(PIPE_IMG, False, True)
        SCREEN.blit(pipe_top, (obs[0], obs[1] - OBSTACLE_GAP // 2 - pipe_top.get_height()))
        # Draw bottom pipe
        SCREEN.blit(PIPE_IMG, (obs[0], obs[1] + OBSTACLE_GAP // 2))

def reset_game():
    global PLAYER_Y, player_velocity, obstacles, score, start_time
    PLAYER_Y = HEIGHT // 2
    player_velocity = 0
    obstacles = [[WIDTH, random.randint(50, HEIGHT - 150), HEIGHT]]
    score = 0
    if start_time is None:
        start_time = datetime.now()
    return np.array([PLAYER_Y, player_velocity, WIDTH, obstacles[0][1]])

def game_step(action):
    global PLAYER_Y, player_velocity, obstacles, score
    player_velocity += GRAVITY
    if action == 1:
        player_velocity = JUMP_STRENGTH
    PLAYER_Y += player_velocity

    # Move and generate obstacles
    for obs in obstacles:
        obs[0] -= config.obstacle_speed
    if obstacles[-1][0] < WIDTH - OBSTACLE_GAP:
        obstacles.append([WIDTH, random.randint(50, HEIGHT - 150), HEIGHT])
    if obstacles[0][0] < -OBSTACLE_WIDTH:
        obstacles.pop(0)

    # Update score
    score += 1

    state = np.array([PLAYER_Y, player_velocity, obstacles[0][0], obstacles[0][1]])
    reward = config.positive_reward if PLAYER_Y > 0 and PLAYER_Y < HEIGHT else config.negative_reward
    done = False if reward > 0 else True

    return state, reward, done

def log_score(score, start_time, end_time):
    duration = end_time - start_time
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([score, start_time.isoformat(), end_time.isoformat(), str(duration)])

def main():
    global PLAYER_Y, player_velocity, score, start_time
    score = 0
    state = reset_game()

    while True:
        SCREEN.blit(BG_IMG, (0, 0))
        CLOCK.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        action = agent.act(state)
        next_state, reward, done = game_step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()  # Train on previous experiences
        state = next_state

        draw_player(PLAYER_X, PLAYER_Y)
        draw_obstacle(obstacles)

        # Display score
        score_text = FONT.render(f"Score: {score}", True, (255, 255, 255))
        SCREEN.blit(score_text, (10, 10))

        if done or score >= 7848:
            end_time = datetime.now()
            print(f"Game Over! Score: {score}")
            log_score(score, start_time, end_time)
            if score >= 7848:
                print(f"Target score reached! Time taken: {end_time - start_time}")
                return
            state = reset_game()
            agent.update_target_model()

        pygame.display.update()

        # Optimize rewards
        if score % 100 == 0:
            config.optimize_rewards()

if __name__ == "__main__":
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Score', 'Start Time', 'End Time', 'Duration'])
    main()
    pygame.quit()
