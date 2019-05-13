import argparse
from tensorboardX import SummaryWriter
import gym
import numpy as np
import time
import torch
from ddpg import DDPG
from normalized_actions import NormalizedActions#----------------------------by default [-1,1]
from ounoise import OUNoise
from param_noise import Adaptive_Parameter_Noise, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from datacenter_adaptive_control_environment import DataCenter_Env
from parameters import Parameters
import Continuous_Cartpole
args = Parameters()
#env = Continuous_Cartpole.ContinuousCartPoleEnv()
env_name = 'MountainCarContinuous-v0'
#env = NormalizedActions(gym.make(env_name))-----------#dont need this coz env.action_space.high returns 1 bound is [-1,1]
env = gym.make(env_name)


#
# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#
# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
# model.learn(total_timesteps=400000)
# model.save("ddpg_mountain")
#
# del model # remove to demonstrate saving and loading
#
# model = DDPG.load("ddpg_mountain")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


#writer = SummaryWriter()

# env.seed(args.seed) Sets the seed for this env's random number generator(s).
# torch.manual_seed(args.seed) You just need to call torch.manual_seed(seed), and it will set the seed of the random number generator to a fixed value,
# so that when you call for example torch.rand(2), the results will be reproducible.
# np.random.seed(args.seed)


agent = DDPG(args.gamma, args.tau, args.actor_hidden_size, env.observation_space.shape[0], env.action_space)

replay_buffer = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[
                      0]) if args.ou_noise else None  # ---------------------------------enable OU noise if passed as argument else discard

param_noise = Adaptive_Parameter_Noise(initial_action_stddev=0.05, desired_action_stddev=args.noise_scale,
                                       adaptation_coefficient=1.05) if args.param_noise else None
# note that the initial_Action_stddev is in terms of parameter space but the desired action std is in terms of the action space

rewards_train = []
rewards_test = []
total_numsteps = 0
global_total_no_of_updates = 0



#Load model if you have a saved model so that you dont have to start from a dumb actor and a dumb critic
# load_pretrained_actor_and_critic = input("Do you want to load a pretrained model? Type \"yes\" or \no\" and press enter")
#
# if load_pretrained_actor_and_critic == "yes":
#
#     try:
#
#         actor_location_to_load = input("Enter the location of the saved actor model.")
#         critic_location_to_load = input("Enter the location of the saved critic.")
#         agent.load_model(actor_location_to_load,critic_location_to_load)
#     except:
#         print("Invalid location given. Starting with a random actor and a random critic.")
#
# else:
#     print("You chose not to load a pretrained actor or a critic.")
#




# ============================================Training
for i_episode in range(args.num_episodes):
    print("Episode No:",i_episode)
    total_numsteps = 0
    state = torch.Tensor([env.reset()])  # -----------------------reset the environment and get the default starting state

    if args.ou_noise:  # ----------------------------------------if OU noise enabled
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise:  # --------------if parameter noise enabled, add noise to the actor's parameters

        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0  # ----------------reward for the episode

    while True:  # -----------------------run the episode until we break by getting done = True after reaching the terminal state

        action = agent.select_action(state, ounoise, param_noise)  # ------------------------>select action using the learning actor
        # print(action.cpu().numpy())
        # time.sleep(222)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])  # ------------------------>returns done value. used by mask as mask = - done,


        # if next state returned is a terminal state then return done = True, hence mask becomes 0  hence V(state before terminal state) = reward + mask * some value
        #env.render()
        total_numsteps += 1
        #print("timestep in the episode: ",total_numsteps)
        episode_reward += reward

        action = torch.Tensor(action.cpu())  # --------------------------convert to Tensor
        mask = torch.Tensor([ not done])  # ------------------------mask is used to make sure that we multiply all the future rewards by 0 at the terminal state
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        replay_buffer.push(state, action, mask, next_state, reward)

        state = next_state  # -------------------------------------now this next state is the new state for which we will take the action acc to the perturbed actor

        # Turns out as soon as we have more 1 element more than the replay batch size in the replay buffer, we update both actor and critic network using
        # update_parameters() at each time step at each episode.
        # Also, at the end of this update_parameters() method exists the soft update for both target actor and target critic

        if len(
                replay_buffer) > args.batch_size:  # ---------------if less elements in replay memory than the batch size chosen, dont do this else do this.

            for _ in range(
                    args.updates_per_step):  # -------Note: We can also du multiple updates even for a single timestep

                transitions = replay_buffer.sample(
                    args.batch_size)  # -------sample a number of transitions from the replay meomory

                batch = Transition(*zip(*transitions))
                #print(batch)

                value_loss, policy_loss = agent.update_parameters(batch)  # ------------>update_parameters() is getting a batch of transitions, returns two loss values

                # writer.add_scalar('loss/value', value_loss,
                #                   global_total_no_of_updates)  # add_scalar(tag, scalar_value, global_step=None, walltime=None)
                # writer.add_scalar('loss/policy', policy_loss, global_total_no_of_updates)

                global_total_no_of_updates += 1
        if done:  # ------------------->if done == True, then break the while loop. Done is the end of this one single action from the actor n/w. we reach next state.
            break

    # writer.add_scalar('reward/train', episode_reward, i_episode)

    # Adapting the param_noise based on distance metric after each episode

    if args.param_noise:
        episode_transitions = replay_buffer.memory_list[replay_buffer.position - total_numsteps:replay_buffer.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
        param_noise.adapt(ddpg_dist)

    rewards_train.append(episode_reward)

    # ==============================================Testing after every 10 episodes
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            #env.render()#--------------------------------------------------removing render to run on the star server
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done:
                break

        # writer.add_scalar('reward/test', episode_reward, i_episode)

        rewards_test.append(episode_reward)
        # Note that this is within this if condition.
        print(
            "Current Episode No: {}, Total numsteps in the last training episode: {}, Testing reward after the last training episode: {}, "
            "Average training reward for the last ten training episodes: {}".format(i_episode, total_numsteps,
                                                                                    rewards_test[-1],
                                                                                    np.mean(rewards_train[-10:])))

# save the actor and the policy that you get after all the episodes
env.render()
agent.save_all_episodes_model(env_name)
env.close()
