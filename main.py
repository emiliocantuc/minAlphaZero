import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Game:

    def initState(self): raise NotImplementedError()

    def nextState(self, state, player, action): raise NotImplementedError()

    def legalActions(self, state, player): raise NotImplementedError()

    def winner(self, board): raise NotImplementedError()

class Connect4(Game):
    
    def __init__(self, rows = 6, cols = 7, players = [1, -1]):
        self.rows = rows
        self.cols = cols
        self.players = players

    def initState(self):
        return torch.zeros((self.rows, self.cols), dtype = torch.int8)
    
    
    def nextState(self, state, player, action):
        new_state = state.clone()
        col = action
        empty_rows = (new_state[:, col] == 0).nonzero(as_tuple=True)[0]
        if len(empty_rows) == 0:
            raise ValueError('Invalid action: column is full')
        row = empty_rows.max().item()

        new_state[row, col] = player
        return new_state

    def legalActions(self, state, player):
        return (state == 0).sum(dim = 0).nonzero(as_tuple = True)[0].tolist()


    def winner(self, board, verbose=False):
        for player in self.players:
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

        return None  # No winner yet
    



if __name__ == "__main__":

    g = Connect4()
    player = 1

    state = g.initState()
    done = False

    while not done:
        print(state)
        legal_actions = g.legalActions(state, player)
        if not legal_actions:
            print("No legal actions available. Game over.")
            break
        
        action = np.random.choice(legal_actions)  # Randomly choose an action for demonstration
        state = g.nextState(state, player, action)
        
        winner = g.winner(state)
        if winner is not None:
            print(f"Player {winner} wins!")
            done = True
        else:
            player *= -1  # Switch players

    print(state)
    g.winner(state, verbose = True)