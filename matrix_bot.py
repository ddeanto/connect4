import time
import numpy as np
from utils import generate_all_segments
from game_board import Connect4Board, Player
import json


ALL_SEGMENTS = generate_all_segments()


class MatrixBot():
    def __init__(self, player: Player, A: np.array = None):
        self.player = player
        self.enemy = Player.O if player == Player.X else Player.X

        self.prev_score = 0

        self.k = len(ALL_SEGMENTS)*2 + 1

        if A is None:
            self.A = 2*(np.random.rand(self.k,2,7) - 0.5)
        else:
            self.A = A

    def mutate(self):
        # TODO: parallelize this
        a,b,c = self.A.shape  

        cells_to_change = 1_000
        for r in np.random.randint(-100, 100+1, cells_to_change):
            x = np.random.randint(0, a)
            y = np.random.randint(0, b)
            z = np.random.randint(0, c)

            self.A[x, y, z] = r

    def board_to_vec(self, connect4: Connect4Board):
        board = connect4.board
        PlayerXs = [(board[*s.T] == Player.X).sum() for s in ALL_SEGMENTS]
        PlayerOs = [(board[*s.T] == Player.O).sum() for s in ALL_SEGMENTS]
        return np.array([self.player] + PlayerXs + PlayerOs)   

    def select_move(self, connect4: Connect4Board):
        moves = [0,0,0,0,0,0,0]
        for n, x in enumerate(self.board_to_vec(connect4)):
            for m in range(7):
                m1, b1 = self.A[n,:,m]
                moves[m] += max(x*m1 + b1, 0)

        moves = list(zip(range(7), moves))
        moves.sort(key=lambda e: e[1], reverse=True)
        for col, value in moves:
            if connect4.column_depth[col] >= 0:
                return col


def genetic_algo(bot1, bot2, count: int=0):
    while True:
        count += 1
        save = count%10==0

        board = Connect4Board()

        while board.winner is None and board.cells_open >= 1:
            if board.whose_turn == bot1.player:
                col = bot1.select_move(board)
            else:
                col = bot2.select_move(board)
            
            board.drop_piece(col)

            # print(board)

        if board.winner is None:
            bot1.mutate()
            bot2.mutate()
        elif Player(board.winner) == bot1.player:
            bot2.mutate()
        elif Player(board.winner) == bot2.player:
            bot1.mutate()

        if save:
            print(f'count: {count}\n{board}')
            with open('A.json', 'w') as f:
                s = json.dumps(bot1.A.tolist())
                f.write(s)   


if __name__ == '__main__':
    bot1 = MatrixBot(player=Player.X)
    bot2 = MatrixBot(player=Player.O)

    genetic_algo(bot1, bot2)
