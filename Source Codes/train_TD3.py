import torch, sys
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from New_datacenter_adaptive_control_environment import DataCenter_Env
max_energy_charging_possible_per_timestep = 0
max_energy_discharging_possible_per_timestep = 0
max_It = max_energy_charging_possible_per_timestep
max_Ot = max_energy_discharging_possible_per_timestep
max_Dt = sys.maxsize #not limititng the value for Dt sinc we ado not know pior to operation what the demand is going to be
def train():
    ######### Hyperparameters #########
    env_name = "Google_Datacenter"
    log_interval = 10           # print avg reward after an interval of
    random_seed = 42
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    learning_rate = 0.001
    exploration_noise = 0.1 
    soft_update_parameter_for_target_actor = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    directory_to_load_pretrained_model_from = "./preTrained/{}".format(env_name) # save trained models

    ###################################
    
    env = DataCenter_Env()
    '''Timeserie values and other variables'''
    state_dim = env.observation_space.shape[0]
    action_dim = 3
    max_action = [float(max_It), float(max_Ot), float(max_Dt)]
    agent = TD3(learning_rate, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    '''Setting a default random seed to repeat an identical run'''
    if random_seed:
        print("Random Seed is set to : {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    '''Logging variables'''
    total_reward_in_29_days = 0
    daily_reward = 0
    log_f = open("log.txt","w+")

    avg_baseline_reward_1_in_29_days = 0
    avg_baseline_reward_2_in_29_days = 0

    days_counter = 1
    
    '''---------------------------------------Training begins----------------------------------------------------------'''
    for day in range(1, max_episodes+1): #29 episodes as max episodes????? No

        state = env.reset()
        for timesteps in range(max_timesteps):
            # select action and add exploration noise:
            action = agent.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])#since TD3 is a deterministic algorithm, add some noise to the action returned by the
                                                                                                    #agent to make sure we explore the action space properly
            #action = action.clip(env.action_space.low, env.action_space.high)
            action = agent.clip_action_custom(action, max_It, max_Ot, max_Dt)
            
            '''Agent takes action in the environment.'''
            next_state, reward, done, _ = env.step(action)
            '''Add the transition to the replay buffer'''
            replay_buffer.add_transition_to_replay_buffer((state, action, reward, next_state, float(done)))
            state = next_state
            
            total_reward_in_episode += reward
            daily_reward += reward
            
            '''If the episode is over, update the policy'''
            if timesteps==(max_timesteps-1):
                agent.update(replay_buffer, timesteps, batch_size, gamma, soft_update_parameter_for_target_actor, policy_noise, noise_clip, policy_delay)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(day, daily_reward))
        log_f.flush()
        daily_reward = 0


        #------------------------------------from here now------------------, see variable counters kata kata halni parne xa and more

        '''For now lets stop training once the monthly cost goes beow greedy or another baseline and save the model.'''
        # if avg cost of the month , not day, less than a certain value right. so reformul;ate the stting such that you > 300 then save and stop traning:

        if total_reward_in_29_days / 29 < avg_baseline_reward_1_in_29_days:
            '''save the model and episode count'''
            print("########## Solved! ###########")
            name = 'Datacenter_solved.'
            agent.save(directory_to_load_pretrained_model_from, name)
            log_f.close()
            break

        if total_reward_in_29_days / 29 < avg_baseline_reward_2_in_29_days:
            '''save the model and episode count'''
            print("########## Solved! ###########")
            name = 'Datacenter_solved.'
            agent.save(directory_to_load_pretrained_model_from, name)
            log_f.close()
            break


        '''Also save the model after each 500 episodes'''
        if day > 500:
            agent.save(directory_to_load_pretrained_model_from, filename)#give proper name here
        
        # print avg reward every log interval:
        if day % log_interval == 0:
            total_reward_in_episode = int(total_reward_in_episode / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(day, total_reward_in_episode))
            total_reward_in_episode = 0

        days_counter = days_counter + 1



if __name__ == '__main__':
    train()
    
