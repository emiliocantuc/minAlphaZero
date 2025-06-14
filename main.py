"""
A toy implementation of AlphaZero. 

At a high level, what happens is:
TODO
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

    def reset(self):
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.visited.clear()

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

        self.Qsa[hstate, best_action] = (Q * Nsa + v) / (Nsa + 1) # rolling average
        self.Nsa[hstate, best_action] = Nsa + 1
        self.Ns[hstate] = Ns + 1

        return -v
    
    def UCT(self, hstate, action):

        Q = self.Qsa[hstate, action]
        Nsa = self.Nsa[hstate, action]
        Ns  = self.Ns[hstate]
        P = self.Ps[hstate].get(action, 0.0)

        return Q + self.c_puct * P * math.sqrt(Ns) / (1 + Nsa)
    
    def improved_policy(self, state, game, net_act_prob, n):

        state = game.state2canonical(state)
        hstate = game.state2str(state)

        # reset visited states
        # TODO do i want to reset everything?
        self.reset()

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

    def __init__(self, net, game):
        self.net = net
        self.game = game

        self.reset_mcts()
    
    def reset_mcts(self):
        self.mcts = MCTS()
    
    @torch.no_grad()
    def act_prob(self, state, temp = 1.0, n = 100):
        # used during self-play as 'target policy'

        # to easily ablate mcts 
        if n > 0:    
            # TODO dont forget temp
            p_improved = self.mcts.improved_policy(state, self.game, self.net.act_prob, n = n)
            return p_improved
        
        else:
            # just use the network to get the prior policy
            state = self.game.state2canonical(state)
            p, _ = self.net.act_prob(state)
            return p
    
    @torch.no_grad()
    def act(self, act_probs, legal_actions = None):
        # just because `choice` sometimes complains probs do not sum to 1
        if legal_actions:
            act_probs = {a: pa for a, pa in act_probs.items() if a in legal_actions}

        p = np.array(list(act_probs.values())).astype(np.float64)
        p = p / np.sum(p)
        return np.random.choice(list(act_probs.keys()), p = p)


def self_play(game, policy, n_games = 1, act_prob_kwargs = {}):
    '''
    Plays policy against itself and collects training examples.
    Each example is a tuple of (canonical_state, policy, value).
    '''

    examples = []

    for i in tqdm(range(n_games), desc='self playing'):

        policy.reset_mcts() # TODO here?
        
        state = game.initState(player = 1 if i % 2 == 0 else -1)
        _examples = []
        
        while not game.gameOver(state):

            p = policy.act_prob(state, **act_prob_kwargs)
            action = policy.act(p)

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

    for epoch in range(10):  # number of epochs
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


def pit(game, p1_fn, pm1_fn, num_games=50):
    results = []
    for i in tqdm(range(num_games), desc='pitting policies'):
        state = game.initState(player = 1 if i % 2 == 0 else -1)
        while not game.gameOver(state):
            board, player = state
            policy = p1_fn if player == 1 else pm1_fn
            action = policy(state, game.legalActions(state))
            state = game.nextState(state, action) # this auto alternates player
        results.append(game.winner(state))

    return np.array(results)

if __name__ == '__main__':

    from connect4 import Connect4, Connect4Net

    device = 'cpu'

    game = Connect4()
    net = Connect4Net(game, device)
    policy = Policy(net, game)

    # import pdb; pdb.set_trace()

    n_episodes = 1#100 
    n_self_play_games = 50

    example_buffer_size = 100_000

    policy_act_prob_kwargs = {
        'temp': 1.0,   # temperature for exploration
        'n': 50,       # number of MCTS simulations
    }

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

        # train the network on the examples
        # train(policy.net, examples)

        # pit the networks against each other and keep best
        old_policy = torch.load('current_policy.pth', weights_only = False)

        pit_results = pit(
            game,
            lambda s, l: policy.act(policy.act_prob(s, **{'temp':1.0, 'n': 50}), l),
            lambda s, l: old_policy.act(old_policy.act_prob(s, **{'temp':1.0, 'n':5}), l),
            num_games = 50 # 100
        )

        print(pit_results.mean())
        print('new policy pit results:')
        print('\t'.join(f'{(pit_results == i).mean():.2f} {l}' for i, l in zip([1,-1,0], ['win','loss','draw'])))
  
        if (pit_results == 1).mean() >= 0.6:
            print('New policy is better, saving...')
            torch.save(policy, 'best_policy.pth')
        else:
            print('Old policy is better, keeping old policy.')
            policy = old_policy