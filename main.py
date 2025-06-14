"""
A toy implementation of AlphaZero. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
from collections import defaultdict

from einops import rearrange

from tqdm import tqdm

# We need a game
class Game:

    def initState(self, player): raise NotImplementedError()

    def nextState(self, state, action): raise NotImplementedError()

    def legalActions(self, state): raise NotImplementedError()

    def winner(self, state): raise NotImplementedError()

    def gameOver(self, state): raise NotImplementedError()

    # TODO view independent of player, used to train network
    def state2canonical(self, state): raise NotImplementedError()

    def state2str(self, state): raise NotImplementedError()

    @property
    def actions(self): raise NotImplementedError


# And Monte Carlo Tree Search
class MCTS:

    def __init__(self, c_puct = 1.0):

        self.c_puct = c_puct

        # (state, action) -> 
        self.Qsa = defaultdict(float)   # 'action/Q value' (expected reward for taking action in state)
        self.Nsa = defaultdict(int)     # number of visits
        # state ->
        self.Ns  = defaultdict(int)     # number of visits
        self.Ps  = defaultdict(dict)    # prior probabilities of actions (action -> prob)

        self.visited = set() # during search

    # def reset(self):
    #     self.Qsa.clear()
    #     self.Nsa.clear()
    #     self.Ns.clear()
    #     self.Ps.clear()
    #     self.visited.clear()

    def search(self, cstate, game, net_act_prob):

        state = cstate # state is canonical, i.e. view independent of player
        hstate = game.state2str(state) # hashable

        # if in terminal state, return value
        if game.gameOver(state):
            return game.winner(state)
        
        # if leaf node, propagate network prediction
        if hstate not in self.visited:
            self.visited.add(hstate)
            self.Ps[hstate], v = net_act_prob(state, game.legalActions(state))
            return -v
        
        # otherwise, select action with UCT
        max_u, best_action = -np.inf, None
        for action in game.legalActions(state):
            u = self.UCT(hstate, action)
            if u > max_u:
                max_u = u
                best_action = action
        
        # take action and recurse
        next_state = game.state2canonical(game.nextState(state, best_action))
        v = self.search(next_state, game, net_act_prob)
    
        # update Q and N values
        Q, Nsa, Ns = self.Qsa[hstate, best_action], self.Nsa[hstate, best_action], self.Ns[hstate]

        # note that we flip v's sign
        self.Qsa[hstate, best_action] = (Q * Nsa - v) / (Nsa + 1) # update rolling average
        self.Nsa[hstate, best_action] = Nsa + 1
        self.Ns[hstate] = Ns + 1

        return v
    
    def UCT(self, hstate, action):

        Q = self.Qsa[hstate, action]
        Nsa = self.Nsa[hstate, action]
        Ns  = self.Ns[hstate]
        P = self.Ps[hstate].get(action, 0.0)

        return Q + self.c_puct * P * math.sqrt(Ns) / (1 + Nsa)
    
    def improved_policy(self, state, game, net_act_prob, n):

        state = game.state2canonical(state)
        hstate = game.state2str(state)

        # note: we reset the tree between moves but more efficient approaches exist
        # e.g. root-shift
        # self.reset()

        # run MCTS search
        for _ in range(n):
            self.search(state, game, net_act_prob)

        # compute policy from visit counts
        total_visits = self.Ns[hstate]
        if total_visits == 0:
            print('No visits recorded, returning None policy.')
            return None

        return {
            a: self.Nsa[hstate, a] / total_visits
            for a in game.actions #legalActions(state)
        }
    

# A player or policy
class PolicyNet(nn.Module):

    def forward(self, cboard):
        pass

    @torch.no_grad()
    def act_prob(self, state, legal_actions) -> dict[str, float]:
        raise NotImplementedError()
    

class Policy:

    def __init__(self, net, game, n, temp = 1.0):
        self.net = net
        self.game = game
        self.n = n
        self.temp = temp
        self.new_game()

    def new_game(self):
        self.mcts = MCTS()
    
    @torch.no_grad()
    def act(self, state):
        # used during self-play as 'target policy'

        if self.n > 0: # to easily ablate mcts  
            p = self.mcts.improved_policy(state, self.game, self.net.act_prob, n = self.n)
            norm = sum(pa ** (1. / self.temp) for pa in p.values())
            p = {a: (pa ** (1. / self.temp)) / norm for a, pa in p.items()} 
        else:
            # just use the network to get the prior policy
            state = self.game.state2canonical(state)
            p, _ = self.net.act_prob(state)

        _p = np.array(list(p.values())).astype(np.float64)
        _p = _p / np.sum(_p)
        action = np.random.choice(list(p.keys()), p = _p)

        return action, p
    
def self_play(game, policy, n_games = 1): 
    '''
    Plays policy against itself and collects training examples.
    Each example is a tuple of (canonical_state, policy, value).
    '''

    examples = []

    for i in tqdm(range(n_games), desc='self playing'):

        policy.new_game()
        state = game.initState(player = 1 if i % 2 == 0 else -1)
        _examples = []
        
        while not game.gameOver(state):

            action, p = policy.act_prob(state)

            # TODO what do we need to return current player for?
            _, player = state
            board, _ = game.state2canonical(state)
            _examples.append((board, player, p, None))

            state = game.nextState(state, action) # this alternates player automatically

        w = game.winner(state)
        examples += [
            (b, pl, p, w * player) # 1 if current player won
            for b, pl, p, _ in _examples
        ]
    
    return examples


def train(net, examples):

    boards   = rearrange([b for b, pl, p, v in examples], 'b h w -> b 1 h w').float()
    target_v = torch.tensor([v for b, pl, p, v in examples]).float()
    target_p = torch.tensor([list(p.values()) for b, pl, p, v in examples]).float()

    dl = DataLoader(
        TensorDataset(boards, target_v, target_p),
        batch_size = 32, shuffle = True
    )

    print(len(dl))

    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    device = net.device

    for epoch in range(5):  # number of epochs
        losses = []
        for b, target_v, target_p in dl:

            b, target_v, target_p = b.to(device), target_v.to(device), target_p.to(device)

            optimizer.zero_grad()
            p, v = net(b)
            
            # calculate loss
            loss_p = F.cross_entropy(p, target_p)
            loss_v = F.mse_loss(v, target_v)
            loss = loss_p + loss_v # TODO regulatization?
            
            # backpropagation
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    
        print(f'Epoch {epoch}, Avg. loss: {np.mean(losses):.4f}')


def pit(game, p1, pm1, num_games=50):
    results = []
    for i in tqdm(range(num_games), desc='pitting policies'):
        p1.new_game(); pm1.new_game()
        state = game.initState(player = 1 if i % 2 == 0 else -1)
        while not game.gameOver(state):
            board, player = state
            policy = p1 if player == 1 else pm1
            action, _ = policy.act(state)
            state = game.nextState(state, action) # this auto alternates player
        results.append(game.winner(state))

    return np.array(results)

if __name__ == '__main__':

    from connect4 import Connect4, Connect4Net

    device = 'cpu'

    game = Connect4()
    net = Connect4Net(game, device)
    policy = Policy(net, game, n = 20, temp = 1.0)

    # import pdb; pdb.set_trace()

    n_episodes = 10#100 
    n_self_play_games = 100

    example_buffer_size = 100_000

    # main loop:
    examples = []
    for _ in range(n_episodes):

        # self play and get training examples
        # examples += self_play(game, policy, n_self_play_games, policy_act_prob_kwargs)

        # if len(examples) > example_buffer_size:
        #     examples = examples[-example_buffer_size:]
        # print(len(examples), 'examples collected')

        # keep copy of the network
        torch.save(policy, 'current_policy.pth')

        # train the network on the self-play examples
        # train(policy.net, examples)

        # pit the networks against each other and keep best
        old_policy = torch.load('current_policy.pth', weights_only = False)

        policy.n = 50
        old_policy.n = 5

        pit_results = pit(game, policy, old_policy, num_games = 50)

        print(pit_results.mean())
        print('new policy pit results:')
        print('\t'.join(f'{(pit_results == i).mean():.2f} {l}' for i, l in zip([1,-1,0], ['win','loss','draw'])))
  
        if (pit_results == 1).mean() >= 0.6:
            print('New policy is better, saving...')
            torch.save(policy, 'best_policy.pth')
        else:
            print('Old policy is better, keeping old policy.')
            policy = old_policy