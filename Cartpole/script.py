import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3                       # Learning Rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500                # Sum of frame at which pole is balanced
score_requirements = 50         # Random game must that have a score greater or equal to 50
initial_games = 10000


def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()          # Generates random action
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games)
    score = 0 
    game_memory = []
    prev_observation = []
