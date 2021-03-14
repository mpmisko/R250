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

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, done, *args):
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
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

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
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return F.leaky_relu(self.head(x.view(x.size(0), -1)))

class DQNAgent:
    def __init__(self, env, test_env, cuda, 
                target_update_interval=10, update_interval=1, start_steps=5000, 
                batch_size=128, gamma=0.9, eps_start=0.95, eps_end=0.05, 
                eps_decay=200, n_actions=5, memory_size=10000):
        self.steps = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.device = 'cpu' if not cuda else 'cuda'
        self.memory = ReplayMemory(memory_size)
        self.policy = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, n_actions).to(self.device)
        self.target = DQNModel(env.obs_window_size + 1, env.obs_window_size + 1, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=0.01)
        self.update_interval = update_interval
        self.start_steps = start_steps
        self.shots = 0
        self.target_update_interval = target_update_interval
        self.train_returns = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy(state.unsqueeze(0).float().to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def exploit(self, state):
        # Act without randomness.
        with torch.no_grad():
            return self.policy(state.unsqueeze(0).float().to(self.device)).max(1)[1].view(1, 1)

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps 

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        
        actions = []
        for i, a in enumerate(batch.action):
            actions.append(a.to(self.device))

        action_batch = torch.stack(actions).squeeze(-1).to(self.device)
        reward_batch = torch.tensor(batch.reward).to(self.device)
        state_action_values = self.policy(state_batch.float()).gather(1, action_batch).squeeze()
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.target(non_final_next_states.float()).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
