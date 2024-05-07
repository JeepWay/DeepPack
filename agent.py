import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from model import doubleDQN
import random

class Agent():
    def __init__(self, args):
        self.args = args
        self.net = doubleDQN(args).to(device=args.device)
        self.target_net = doubleDQN(args).to(device=args.device)
        self.init_target_net()
        self.net.train()
        self.target_net.train()
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=((1-(1/args.iterations))))
        self.loss_fn = torch.nn.MSELoss()


    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return self.net(state).argmax(1).item()


    def learn(self, replay):
        minibatch = random.sample(replay, self.args.batch_size)
        state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(self.args.device)
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.args.device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.args.device)
        state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(self.args.device)
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.args.device)

        # Get Q values
        q_values = self.net(state1_batch.float()).gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

        # Get target Q values
        with torch.no_grad():
            max_action = self.net(state2_batch.float()).argmax(1)
            max_next_q_values = self.target_net(state2_batch.float()).gather(1, max_action.unsqueeze(dim=1)).squeeze()
            target_q_values = reward_batch + (self.args.gamma * max_next_q_values * (1 - done_batch))

        # Calculate loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def init_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def save(self, path, name='model.pt'):
        torch.save(self.net.state_dict(), os.path.join(path, name))

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


