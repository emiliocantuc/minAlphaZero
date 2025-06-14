import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
from collections import defaultdict

from einops import rearrange

from main import Game, PolicyNet

class Connect4(Game):

    # state is (2d tensor with board, player)
    # action is the column to drop the piece in
    
    def __init__(self, rows = 6, cols = 7):
        self.rows = rows
        self.cols = cols

    def initState(self, player = 1):
        return torch.zeros((self.rows, self.cols), dtype = torch.int8), player
    
    def nextState(self, state, action):
        board, player = state
        new_board = board.clone()

        col = action
        empty_rows = (new_board[:, col] == 0).nonzero(as_tuple=True)[0]
        if len(empty_rows) == 0: raise ValueError('Invalid action: column is full')
        row = empty_rows.max().item()

        new_board[row, col] = player
        return new_board, -player

    def legalActions(self, state):
        board, _ = state
        return (board == 0).sum(dim = 0).nonzero(as_tuple = True)[0].tolist()

    def winner(self, state, verbose=False):
        board, _ = state
        for player in [-1, 1]:
            # Horizontal check
            for r in range(self.rows):
                for c in range(self.cols - 3):
                    if all(board[r, c+i] == player for i in range(4)):
                        if verbose:
                            print(f"Player {player} wins horizontally at row {r}, columns {c}-{c+3}")
                        return player

            # Vertical check
            for c in range(self.cols):
                for r in range(self.rows - 3):
                    if all(board[r+i, c] == player for i in range(4)):
                        if verbose:
                            print(f"Player {player} wins vertically at column {c}, rows {r}-{r+3}")
                        return player

            # Diagonal (top-left to bottom-right)
            for r in range(self.rows - 3):
                for c in range(self.cols - 3):
                    if all(board[r+i, c+i] == player for i in range(4)):
                        if verbose:
                            print(f"Player {player} wins diagonally (\\) starting at row {r}, column {c}")
                        return player

            # Diagonal (bottom-left to top-right)
            for r in range(3, self.rows):
                for c in range(self.cols - 3):
                    if all(board[r-i, c+i] == player for i in range(4)):
                        if verbose:
                            print(f"Player {player} wins diagonally (/) starting at row {r}, column {c}")
                        return player

        # No winner yet
        if len(self.legalActions(state)) == 0:
            if verbose:
                print("Game over: Draw")
            return 0
        
        return None
    
    def gameOver(self, state):
        return len(self.legalActions(state)) == 0 or self.winner(state) is not None
    
    def state2canonical(self, state):
        # always return from player 1's pov
        board, player = state
        return board * player, 1
    
    def state2str(self, state):
        board, player = state
        return str(board.numpy()), player # str is enough here since board is small

    @property
    def actions(self): return list(range(self.cols))


# We demonstrate with this but you can add any other game
class Connect4Net(nn.Module): # TODO subclass PolicyNet?

    def __init__(self, game, device):
        super(Connect4Net, self).__init__()

        self.game = game
        self.device = device
        
        # TODO make this more flexible? other board sizes?
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 filters of size 3x3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 filters of size 3x3
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),  # Fully connected layer
            nn.ReLU()
        )

        self.p_head = nn.Linear(128, 7)  # Output layer for policy (7 actions)
        self.v_head = nn.Linear(128, 1)

        self.to(self.device)

    def forward(self, x):
        h = self.backbone(x.float())
        return self.p_head(h), torch.tanh(self.v_head(h)).squeeze(-1)
    
    @torch.no_grad()
    def act_prob(self, state, legalActions):
        # used during MTCS search to get the prior policy and value

        self.eval()

        board, player = self.game.state2canonical(state)
        # board should always be from player 1's pov
        assert player == 1 and (board == 1).sum() <= (board == -1).sum()

        x = rearrange(board, 'h w -> 1 1 h w').float().to(self.device)
        p, v = self(x)
        p = rearrange(p, '1 a -> a')

        mask = torch.ones_like(p, dtype = torch.bool) # TODO simplify this
        mask[legalActions] = False
        p[mask] = float('-inf')  # disregard invalid actions

        p = F.softmax(p, dim = -1)  
        return dict(zip(range(7), p.cpu().numpy().tolist())), v.cpu()
    
   
def board_to_string(board):
    symbol_map = {0: '.', 1: 'X', -1: 'O'}
    lines = []
    for row in board:
        line = ' '.join(symbol_map[int(cell)] for cell in row)
        lines.append(line)
    lines.append('-' * (2 * board.shape[1] - 1))  # Separator line
    lines.append(' '.join(str(i) for i in range(board.shape[1])))  # Column indices
    return '\n'.join(lines)

def play_from_terminal(policy):
    # 1 is user, -1 is AI
    game = Connect4()
    player = 1 if np.random.rand() < 0.5 else -1
    state = game.initState(player)

    while not game.gameOver(state):
        print("Current board:")
        print(board_to_string(state[0]))
        print(f"Player {state[1]}'s turn")

        if state[1] == 1:
            action = int(input("Enter column (0-6): "))
            if action not in game.legalActions(state):
                print("Invalid action. Try again.")
                continue
        else:
            legal_actions = game.legalActions(state)
            action_probs = policy.act_prob(game.state2canonical(state), legal_actions)
            action = policy.act(action_probs, legal_actions)

        state = game.nextState(state, action)
    game.winner(state, verbose=True)
    print("Final board:")
    print(board_to_string(state[0]))

if __name__ == "__main__":

    from main import Policy, MCTS

    policy = torch.load('best_policy.pth', weights_only = False)
    play_from_terminal(policy)

    # Example usage
    # game = Connect4()
    # net = Connect4Net()

    # # Random game
    # state = game.initState()
    # i = 1
    # while not game.gameOver(state):
    #     legal_actions = game.legalActions(state)
    #     print(state)
    #     action_probs, value = net.act_prob(game.state2canonical(state), legal_actions)
    #     p = np.array(list(action_probs.values())).astype(np.float64)
    #     p = p / np.sum(p)
    #     action = np.random.choice(list(action_probs.keys()), p=p)
    #     state = game.nextState(state, action)
    #     print(i)
    #     i += 1

    # winner = game.winner(state, verbose=True)
    # print(winner)


    