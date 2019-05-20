import sys
import keyboard
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import os

actor_model_name_to_save = 'trained_actor_model.pth'
critic_model_name_to_save = "trained_critic_model.pth"


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

"""
From: https://github.com/pytorch/pytorch/issues/1959
There's an official LayerNorm implementation in pytorch now, but it hasn't been included in 
pip version yet. This is a temporary version
This slows down training by a bit
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)
        self.cuda()

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))#returns actions between -1 and 1 by default
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)
        self.cuda()


    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V

class DDPG(object):
    counter = 0
    def __init__(self, gamma, tau, actor_hidden_size, num_inputs, action_space): #hidden size is the no of nodes in each layer of the actor

        self.num_inputs = num_inputs
        self.action_space = action_space#output dimension of the action

        self.actor = Actor(actor_hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_target = Actor(actor_hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_perturbed = Actor(actor_hidden_size, self.num_inputs, self.action_space).to(device)#actor with the parameter noise added
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(actor_hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_target = Critic(actor_hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-2)

        self.gamma = gamma #discounting factor
        self.tau = tau#target update factor

        hard_update(self.actor_target, self.actor)  #initializing the target networks with the same parameters as the learning network
        hard_update(self.critic_target, self.critic)



    def select_action(self, state, action_noise=None, param_noise=None):

        self.actor.eval()#-------------------------------------------------actor in the evaluation mode
        #self.actor_perturbed.eval()

        if param_noise is not None:
            # import time
            # time.sleep(11111)
            mu = self.actor_perturbed(state)#------------------action given by the perturned learning actor
        else:

            state = Variable(state, volatile=False, requires_grad=False).cuda()

            mu = self.actor(state)
        self.actor.train()  # changing back to the training mode
        #self.actor_perturbed.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).cuda()

        #print("unclamped action is ",mu) is between -1 and 1
        return mu.clamp(-1, 1)#-----------------------------clamping the actions in between -1 and 1 as [-1,1] range


    def update_parameters(self, batch):#---------------------->getting a batch of sample transitions from the replay buffer

        save_flag = True#Flag for saving actor and critic models
        #print("here")
        # state_batch = Variable(torch.cat(batch.state).cuda())#--------putting together all the states in the batch data provoded as the argument
        # action_batch = Variable(torch.cat(batch.action).cuda())#------putting together all the actions taken for those states
        # reward_batch = Variable(torch.cat(batch.reward).cuda())
        # mask_batch = Variable(torch.cat(batch.mask).cuda())#----------yet to understand its purpose
        # next_state_batch = Variable(torch.cat(batch.next_state).cuda())
        state_batch = torch.cat(batch.state).cuda()  # --------putting together all the states in the batch data provoded as the argument
        action_batch =torch.cat(batch.action).cuda() # ------putting together all the actions taken for those states
        reward_batch = torch.cat(batch.reward)
        mask_batch = torch.cat(batch.mask)  # ----------yet to understand its purpose
        next_state_batch = torch.cat(batch.next_state)
        next_state_batch = Variable(next_state_batch, volatile=False, requires_grad=False).cuda()


        #what actions would the target actor give for the NEXT STATES (not the states) that the learning actor saw
        # print("cc",self.actor_target(next_state_batch))
        # import time
        # time.sleep(222)
        next_action_batch = self.actor_target(next_state_batch)
        #what q values would the target critic give to the same <NEXT STATES,actions generated by target actor> pairs
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        #the following is very subtle and important
        #what would be the action values for the starting states had we started at the started states, taken the action determined by the learning actor to
        #get the rewards as per the reward_batch and then take actions generated by the target_actor for the next state.

        import time
        expected_state_action_values_batch = reward_batch.cpu() + (self.gamma * mask_batch.cpu() * next_state_action_values.cpu())

        #we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # This is convenient while training RNNs. So, the default action is to accumulate the gradients on every loss.backward() call.
        #Basically, just do zero_grad() before doing backprop in pytorch

        self.critic_optimizer.zero_grad()


        state_action_batch = self.critic((state_batch), (action_batch))#what would be the action values for the starting states had we used the learning actor
                                                                        #and Q networks

        value_loss = F.mse_loss(state_action_batch.cpu(), expected_state_action_values_batch)#what is the difference between the learning and the target actors and critics

        value_loss.backward()#loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
                            # These are accumulated into x.grad for every parameter x

        self.critic_optimizer.step()

        # In case keyboard 's' is pressed
        # if keyboard.is_pressed('s') and save_flag:
        #     # Save flag is marked as False
        #     save_flag = False
        #     # save actor and critic
        #     suffix_to_add_on_saved_model_name = input("Enter the suffix to add to the model name.")
        #     self.save_interrupted_model(suffix_to_add_on_saved_model_name)
        #
        # # This helps to prevent saving the model multiple times accidentally
        # if (not save_flag and not (keyboard.is_pressed('s'))):
        #     # Save flag is marked as True
        #     save_flag = True

        self.actor_optimizer.zero_grad()

        #For the policy network, the task was to find the best action that would have returned the
        # maximum q value for the state input to it. Hence, our objective would be to maximize the expected return.
        #To calculate the policy loss, we take the derivative of the objective function with respect to the parameters of the learning policy network

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        #why negative of the critic up there? because we want to maximize the cumulative return but optimizing
        #loss means we wanna minimize. So changing the sign to minimize the negative of the expected return.


        policy_loss = policy_loss.mean()# we take the mean of the sum of the gradients since we are taking mini-batches
        policy_loss.backward()
        self.actor_optimizer.step()

        # In case keyboard 's' is pressed
        # if keyboard.is_pressed('s') and save_flag:
        #     # Save flag is marked as False
        #     save_flag = False
        #     # save actor and critic
        #     suffix_to_add_on_saved_model_name = input("Enter the suffix to add to the model name.")
        #     self.save_interrupted_model(suffix_to_add_on_saved_model_name)
        #
        #
        #
        # # This helps to prevent saving the model multiple times accidentally
        # if (not save_flag and not (keyboard.is_pressed('s'))):
        #     # Save flag is marked as True
        #     save_flag = True

        #soft update the target networks toward the learning networks for stable learning
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise): #param_noise has initial_action_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01
        """Apply parameter noise to actor model, for exploration"""

        hard_update(self.actor_perturbed, self.actor)#hard_update(target, source)--> set parameters of the perturbed actors
        params = self.actor_perturbed.state_dict()#--get the parameters of the perturbed actor
        for name in params:
            if 'ln' in name:
                pass #null operation if the parameter has "ln" as its name#-------------------------Ask Shauharda dai
            param = params[name]
            x = torch.tensor(param_noise.current_stddev).cuda()

            param += torch.tensor(torch.randn(param.shape)).cuda() * x# multiply each parameter with the adaptive standard deviation

    def save_all_episodes_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models_after_all_episodes/'):
            os.makedirs('models_after_all_episodes/')

        if actor_path is None:
            actor_path = "models_after_all_episodes/ddpg_actor_{}_after_{}_episodes.pth".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models_after_all_episodes/ddpg_critic_{}_{}.pth".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def save_interrupted_model(self, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models_after_keyboard_interruption/'):
            os.makedirs('models_after_keyboard_interruption/')

        if actor_path is None:
            actor_path = "models_after_keyboard_interruption/ddpg_actor_{}_{}.pth".format(suffix)
        if critic_path is None:
            critic_path = "models_after_keyboard_interruption/ddpg_critic_{}_{}.pth".format(suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)


    def load_model(self, actor_path, critic_path):
        try:

            print('Loading models from {} and {}'.format(actor_path, critic_path))
            if actor_path is not None:
                self.actor.load_state_dict(torch.load(actor_path))
            if critic_path is not None:
                self.critic.load_state_dict(torch.load(critic_path))
            print("Models loaded successfully!!!")
        except:
            print("Could not load models!!!")
