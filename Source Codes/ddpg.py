import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


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

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))
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
    def __init__(self, gamma, tau, actor_hidden_size, num_inputs, action_space): #hidden size is the no of nodes in each layer of the actor

        self.num_inputs = num_inputs
        self.action_space = action_space#output dimension of the action

        self.actor = Actor(actor_hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(actor_hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(actor_hidden_size, self.num_inputs, self.action_space)#actor with the parameter noise added
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(actor_hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(actor_hidden_size, self.num_inputs, self.action_space)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma #discounting factor
        self.tau = tau#target update factor

        hard_update(self.actor_target, self.actor)  #initializing the target networks with the same parameters as the learning network
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None, param_noise=None):

        self.actor.eval()#-------------------------------------------------actor in the evaluation mode
        if param_noise is not None:
            mu = self.actor_perturbed((Variable(state)))#------------------action given by the perturned learning actor
        else:
            mu = self.actor((Variable(state)))

        self.actor.train()#changing back to the training mode
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu.clamp(-1, 1)#-----------------------------clamping the actions in between -1 and 1 as [-1,1] range


    def update_parameters(self, batch):#---------------------->getting a batch of sample transitions from the replay buffer

        state_batch = Variable(torch.cat(batch.state))#--------putting together all the states in the batch data provoded as the argument
        action_batch = Variable(torch.cat(batch.action))#------putting together all the actions taken for those states
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))#----------yet to understand its purpose
        next_state_batch = Variable(torch.cat(batch.next_state))

        #what actions would the target actor give for the NEXT STATES (not the states) that the learning actor saw
        next_action_batch = self.actor_target(next_state_batch)
        #what q values would the target critic give to the same <NEXT STATES,actions generated by target actor> pairs
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        #the following is very subtle and important
        #what would be the action values for the starting states had we started at the started states, taken the action determined by the learning actor to
        #get the rewards as per the reward_batch and then take actions generated by the target_actor for the next state.

        expected_state_action_values_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        #we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # This is convenient while training RNNs. So, the default action is to accumulate the gradients on every loss.backward() call.
        #Basically, just do zero_grad() before doing backprop in pytorch

        self.critic_optimizer.zero_grad()


        state_action_batch = self.critic((state_batch), (action_batch))#what would be the action values for the starting states had we used the learning actor
                                                                        #and Q networks

        value_loss = F.mse_loss(state_action_batch, expected_state_action_values_batch)#what is the difference between the learning and the target actors and critics

        value_loss.backward()#loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
                            # These are accumulated into x.grad for every parameter x

        self.critic_optimizer.step()

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
            param += torch.randn(param.shape) * param_noise.current_stddev # multiply each parameter with the adaptive standard deviation

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))