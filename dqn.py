from collections import namedtuple
import random
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
import utils
from tqdm import tqdm

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
steps_done = 0
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_actions = 8

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNModel(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return F.leaky_relu(self.head(x.view(x.size(0), -1)))

def select_action(state, policy):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def optimize_model(memory, policy, target, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)

    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action).squeeze(-1)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(env, args, is_rendering):
    global steps_done
    
    agent1_rewards = []
    agent2_rewards = []
    memory1 = ReplayMemory(10000)
    memory2 = ReplayMemory(10000)

    policy1 = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, 8)
    policy2 = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, 8)
    
    target1 = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, 8)
    target1.load_state_dict(policy1.state_dict())
    target1.eval()

    target2 = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, 8)
    target2.load_state_dict(policy2.state_dict())
    target2.eval()

    optimizer1 = optim.RMSprop(policy1.parameters(), lr=0.01)
    optimizer2 = optim.RMSprop(policy2.parameters())

    for i_episode in range(args.num_episodes):

        print(f"Starting training episode: {i_episode + 1} ... ")
        curr_epi_steps = 0

        # Initialize the environment and state
        env.reset()
        state = env.get_observations()
        next_torch_state1 = torch.transpose(torch.tensor(state[0].copy()), 0, -1).float()
        next_torch_state2 = torch.transpose(torch.tensor(state[1].copy()), 0, -1).float()
        
        for t in tqdm(range(args.exp_steps)):
            steps_done += 1
            curr_epi_steps += 1

            # Select and perform an action
            torch_state1 = next_torch_state1
            torch_state2 = next_torch_state2
            action1 = select_action(torch_state1, policy1)
            action2 = select_action(torch_state2, policy2)
            if is_rendering or i_episode > int(args.num_episodes * 0.95):
                env.render()

            next_state, rewards, dones, _ = env.step([action1.item(), action2.item()])
            
            reward1 = torch.tensor([rewards[0]]).float()
            reward2 = torch.tensor([rewards[1]]).float()
            agent1_rewards.append(reward1.item())
            agent2_rewards.append(reward2.item())
            
            if all(dones):
                next_torch_state1 = None
                #next_torch_state2 = None
            else:
                next_torch_state1 = torch.transpose(torch.tensor(next_state[0].copy()), 0, -1).float()
                #next_torch_state2 = torch.transpose(torch.tensor(next_state[1].copy()), 0, -1).float()

            # Store the transition in memory
            memory1.push(torch_state1, action1, next_torch_state1, reward1)
            #memory2.push(torch_state2, action2, next_torch_state2, reward2)

            # Perform one step of the optimization (on the target network)
            optimize_model(memory1, policy1, target1, optimizer1)
            #optimize_model(memory2, policy2, target2, optimizer2)

            if all(dones):
                break
    
        print(f"Average episode reward (agent1): {sum(agent1_rewards[-curr_epi_steps:]) / curr_epi_steps}")
        print(f"Average episode reward (agent2): {sum(agent2_rewards[-curr_epi_steps:]) / curr_epi_steps}")

        utils.plot_stats(agent1_rewards, agent2_rewards, i_episode)
        if i_episode % TARGET_UPDATE == 0:
            target1.load_state_dict(policy1.state_dict())
            target2.load_state_dict(policy2.state_dict())