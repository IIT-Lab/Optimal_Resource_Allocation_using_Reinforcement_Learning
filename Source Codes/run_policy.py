import gym
from tensorboardX import SummaryWriter

import numpy as np
import time
import torch
from ddpg import DDPG
from normalized_actions import NormalizedActions
from parameter_noise import Adaptive_Parameter_Noise, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from datacenter_adaptive_control_environment import DataCenter_Env
from parameters import Parameters


args = Parameters()
env_name = 'MountainCarContinuous-v0'
#env = NormalizedActions(gym.make(env_name))
env = gym.make(env_name)

writer = SummaryWriter()
agent = DDPG(args.gamma, args.tau, args.actor_hidden_size, env.observation_space.shape[0], env.action_space)

replay_buffer = ReplayMemory(args.replay_size)


rewards_train = []
rewards_test = []
total_numsteps = 0
global_total_no_of_updates = 0


#Load model
load_pretrained_actor_and_critic = input("Do you want to load a pretrained model? Type \"yes\" or \no\" and press enter")

if load_pretrained_actor_and_critic == "yes":

    try:

        actor_location_to_load = input("Enter the location of the saved actor model.")
        critic_location_to_load = input("Enter the location of the saved critic.")
        agent.load_model(actor_location_to_load,critic_location_to_load)
    except:
        print("Invalid location given. Starting with a random actor and a random critic.")

elif load_pretrained_actor_and_critic == "no":
    print("You chose not to load a pretrained actor or a critic.")
    exit()

else:
    print("Invalid Input")
    exit()#----------------------------------------------------------------If invalid Input, exit.


for i_episode in range(args.num_episodes):

        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.numpy()[0])
            env.render()#--------------------------------------------------adding render() to run on the star server
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done:
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        rewards_test.append(episode_reward)
        # Note that this is within this if condition.
        print("Running the loaded policy---\nCurrent Episode No: {}\n Current episode's reward: {}\n, ".format(i_episode, episode_reward))


env.close()
