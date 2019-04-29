import argparse
from tensorboardX import SummaryWriter

import numpy as np

import torch
from ddpg import DDPG
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import Adaptive_Parameter_Noise, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='DDPG')


parser.add_argument('--gamma', type=float, default=0.9, metavar='G', #metavar : A name for the argument in usage messages.
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target network update factor, (default: 0.001)')

#  The weights of target networks are updated by having them slowly track the learned networks:
#  θ​ = τθ+(1−τ)θ where τ≪1.
#  This means that the target values are constrained to change slowly, greatly improving the stability of learning.


parser.add_argument('--ou_noise', type=bool, default=False)#--------------------------------------Need to check on this
parser.add_argument('--param_noise', type=bool, default=True)#--------------------------------------Need to check on this
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of nodes in all the layers of the actor except the op layer (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',#-------------------------doing 5 updates for the networks even for a single time step
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',#------------------------size of the replay buffer
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()


#env = NormalizedActions(gym.make(args.env_name))
writer = SummaryWriter()



env = None#--------------------------------------------For now, no environment specified.



#env.seed(args.seed) Sets the seed for this env's random number generator(s).
#torch.manual_seed(args.seed) You just need to call torch.manual_seed(seed), and it will set the seed of the random number generator to a fixed value,
# so that when you call for example torch.rand(2), the results will be reproducible.
#np.random.seed(args.seed)


agent = DDPG(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space)

replay_buffer = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None#---------------------------------Need to check

param_noise = Adaptive_Parameter_Noise(initial_stddev=0.05, desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

rewards = []
total_numsteps = 0
global_total_no_of_updates = 0


#============================================Training
for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])#-----------------------reset the environment and get the default starting state

    if args.ou_noise: #----------------------------------------if OU noise enabled
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise:#--------------if parameter noise enabled
        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0#----------------reward for the episode

    while True:#-----------------------run the episode until we break by getting done = True after reaching the terminal state

        action = agent.select_action(state, ounoise, param_noise)#------------------------>select action using the learning actor
        next_state, reward, done, _ = env.step(action.numpy()[0])#------------------------>returns done value used as a mask, done is a Boolean Value

        total_numsteps += 1
        episode_reward += reward

        action = torch.Tensor(action)#--------------------------convert to Tensor
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        replay_buffer.push(state, action, mask, next_state, reward)

        state = next_state#-------------------------------------now this next state is the new state for which we will take the action acc to the perturbed actor


        #Turns out as soon as we have more 1 element more than the replay batch size in the replay buffer, we update both actor and critic network using
        #update_parameters() at each time step at each episode.
        # Also, at the end of this update_parameters() method exists the soft update for both target actor and target critic

        if len(replay_buffer) > args.batch_size: #---------------if less elements in replay memory than the batch size chosen, dont do this else do this.

            for _ in range(args.updates_per_step):#-------Note: We can also du multiple updates even for a single timestep

                transitions = replay_buffer.sample(args.batch_size)#-------sample a number of transitions from the replay meomory

                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)#------------>update_parameters() is getting a batch of transitions, returns two loss values

                writer.add_scalar('loss/value', value_loss, global_total_no_of_updates)# add_scalar(tag, scalar_value, global_step=None, walltime=None)
                writer.add_scalar('loss/policy', policy_loss, global_total_no_of_updates)

                global_total_no_of_updates += 1
        if done:#------------------->if done == True, then break the while loop. Done is the end of this one single action from the actor n/w. we reach next state.
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)

    # adapting the param_noise based on distance metric

    if args.param_noise:
        episode_transitions = replay_buffer.memory_list[replay_buffer.position - t:replay_buffer.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
        param_noise.adapt(ddpg_dist)

    rewards.append(episode_reward)


    #==============================================Testing after every 10 episodes
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done:
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
    
env.close()
