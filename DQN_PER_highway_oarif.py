import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import highway_env
import gym
import random

# set environment and get dimension parameters
env = gym.make("highway-fast-v0")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

### Constants for training
learning_rate = 5e-4
epochs = 2000
batch_size = 64
epsilon = 0.1

##########################

#############################################
####### BUILDING A NEURAL NETWORK ###########
##### REPRESENTING ACTION STATE VALUES ######
#############################################

# net_Qvalue is a neural network representing an action state value function:
# it takes as inputs observations and outputs values for each action
net_Qvalue = nn.Sequential(
        nn.Linear(obs_dim**2, 32), 
        nn.Tanh(),
        nn.Linear(32, n_acts)
        )

# net_Qvalue_target is another one
net_Qvalue_target = nn.Sequential(
        nn.Linear(obs_dim**2, 32), 
        nn.Tanh(),
        nn.Linear(32, n_acts)
        )
net_Qvalue_target.eval()


### In prioritized experience replay, we save all sampled experiences in a buffer (a form of memory or history) and we randomly pick a number of experiences
### from the buffer for training (in order to break the correlation between the successive experiences to have an unbiased learning). The priorities of the 
### experiences are saved and updated at each gradient update. Those priorities consist of weights in which the random picking of samples is made. 

experience_buffer = [] # buffer of experiences
priorities = [] # priorities of each experience in the buffer

# constants used for experience replay
buffer_size = 5000
offset = 1e-2
priority_value = 0.8   # between 0 and 1
sampling_importance = 0.4 # used to correct bias resulting from choosing the more prioritized experiences during training over and over


### Helper functions to use for prioritized experience replay

# Function to add an experience and its priority to the buffer 
def add_exp(exp, prio):
    # if the length of the buffer is inferior to the buffer_size, we add the experience to the buffer
    # or else we remove the first element of the buffer then add the new experience (and its priority)
    if len(experience_buffer) < buffer_size:
        experience_buffer.append(exp)
        priorities.append(prio)
        priorities[len(experience_buffer)-1] /= sum(priorities) # for normalisation
    else:
        experience_buffer.pop(0)
        experience_buffer.append(exp)
        priorities.pop(0)
        priorities.append(prio)
        priorities[len(experience_buffer)-1] /= sum(priorities) # for normalisation

# Function to get the priority of a newly sampled experience (likelyhood of the experience to get picked for training)
def get_priority(exp):
    with torch.no_grad():
        # the priority of the experience is set to be proportional to the TD error because it indicates its rarety
        error = abs(net_Qvalue(torch.as_tensor(exp[0], dtype=torch.float32)).max(0)[1].item() 
                - (exp[2] + net_Qvalue_target(torch.as_tensor(exp[3], dtype=torch.float32)).max(0)[1].item()))
        # we use an offset term to avoid having a priority equal to 0 (in case the error is 0) and we elevate to the priority_value'th power 
        # if priority_value=1 then its fully prioritized experience replay 
        # and if priority_value=0 then the experience replay is without priority (we pick uniformly from the buffer) as priorities are all set to 1
    return (error + offset) ** priority_value

# Function to update priorities of experiences based on their errors and their index in the buffer
def update_priorities(error, indices):
    for count, index in enumerate(indices):
        with torch.no_grad():
            priorities[index] = (abs(error[count]) + offset) ** priority_value 
        priorities[index] /= sum(priorities)

# function to update the sampling_importance exponent in order to reach 1 at the end of the training
def update_sampling_importance(s):
    s += (1-s)/epochs

def choose_action(observation):
    if random.random() < epsilon:
        return random.randrange(n_acts)
    else:
        with torch.no_grad():
            q_values = net_Qvalue(observation)
            return q_values.max(0)[1].item()

def compute_loss(batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_non_final, importance_weights):
    batch_q_values = net_Qvalue(batch_observations)
    # print("Qmean:", torch.mean(batch_q_values.detach()))
    batch_q_value = batch_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

    batch_q_value_next = torch.zeros_like(batch_q_value)
    with torch.no_grad():
        next_non_final_observations = batch_next_observations[batch_non_final]
        batch_q_values_next = net_Qvalue(next_non_final_observations)
        _, batch_max_indices = batch_q_values_next.max(dim=1)
        batch_q_values_next = net_Qvalue_target(next_non_final_observations)
        batch_q_value_next[batch_non_final] = batch_q_values_next.gather(1, batch_max_indices.unsqueeze(1)).squeeze(1)

    batch_expected_q_value = batch_rewards + batch_q_value_next

    error = batch_q_value - batch_expected_q_value
    # multiply element wise batch loss terms with importance weights as the step() method does not support a weighted gradient update for a batch sample
    loss = (error).pow(2) * importance_weights
    loss = loss.mean()
    # We return both the error and the loss since the loss will be used in the optimization step and the error will be used to update the priorities
    # of the experiences picked in the batch
    return loss, error

# make optimizer
optimizer = Adam(net_Qvalue.parameters(), lr = learning_rate)


def DQN():
    for i in range(epochs):
        update_sampling_importance(sampling_importance)
        # we copy the parameters of Qvalue into Qvalue_target every 20 iterations
        if i % 20 == 0:
            net_Qvalue_target.load_state_dict(net_Qvalue.state_dict())

        batch_observations = [] 
        batch_actions = []      
        batch_rewards = []      
        batch_next_observations = [] 
        batch_non_final = []
        importance_weights = []

        # for statistics over all episodes run in the first step
        episodes = 0
        total_reward = 0
        episode_rewards = []

        # reset episode-specific variables
        observation = env.reset()
        done = False

        # First step: collect experience by simulating the environment using the current policy
        for j in range (batch_size):
            old_observation = observation.copy()
            # flatten the observation array in order to get appropriate input for the neural networks because the observation space is of shape (5,5)
            action = choose_action(torch.as_tensor(old_observation.flatten(), dtype=torch.float32))

            # sample one step from the environment
            observation, reward, done, _ = env.step(action)

            experience = (old_observation.flatten(), action, reward, observation.flatten(), done)           
            priority = get_priority(experience)
            # add the sampled experience and its priority the buffer
            add_exp(experience, priority)

            
            total_reward += reward
            if done:
                observation = env.reset()
                episodes += 1
                episode_rewards.append(total_reward)
                total_reward = 0 

        # randomly pick batch_size experiences from the buffer with respect to their priorities which represent the distribution of probabilities
        # of picking these experiences
        batch_sample_indices = random.choices(list(range(len(experience_buffer))), weights=priorities, k=batch_size)

        for k in batch_sample_indices:
            batch_observations.append(experience_buffer[i][0])
            batch_actions.append(experience_buffer[i][1])
            batch_rewards.append(experience_buffer[i][2])
            batch_next_observations.append(experience_buffer[i][3])
            batch_non_final.append(not experience_buffer[i][4])
            # weights used to make the update step smaller for more prioritized experiences as they are picked more often than others
            # the sampling_importance exponent gets bigger as the training advances because the bias resulteing from the frequent picking becomes bigger
            importance_weights.append(1/(len(experience_buffer)*priorities[k])**sampling_importance)  


            

        # Second step: update the policy
        # we take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss, batch_error = compute_loss(torch.as_tensor(np.array(batch_observations), dtype=torch.float32),
                                  torch.as_tensor(np.array(batch_actions), dtype=torch.int64),
                                  torch.as_tensor(np.array(batch_rewards), dtype=torch.float32),
                                  torch.as_tensor(np.array(batch_next_observations), dtype=torch.float32),
                                  torch.as_tensor(np.array(batch_non_final), dtype=torch.bool),
                                  torch.as_tensor(np.array(importance_weights), dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()

        # update the priorities of the experiences present in the batch after a gradient update step
        update_priorities(batch_error, batch_sample_indices)


        if i % 10 == 0:
            mean_reward = np.mean(episode_rewards)
            print('epoch: %3d \t loss: %.3f \t mean_reward: %3d' % (i, batch_loss, mean_reward))

DQN()

###### EVALUATION ############

def run_episode(env, render = False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(torch.as_tensor(obs.flatten(), dtype=torch.float32))
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

epsilon = 0
policy_scores = [run_episode(env) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

for _ in range(2):
  run_episode(env, True)

env.close()
