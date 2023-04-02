from matrix_bot import MatrixBot
from game_board import Connect4Board, Player
import time
import json
import numpy as np


def play_bot():
    board = Connect4Board()

    A = np.array(json.loads(open('/Users/darrendeanto/workspace/connect4/A.json').read()))
    trained_bot = MatrixBot(player=Player.O, A=A)

    while not board.winner:
        if board.whose_turn == trained_bot.player:
            col_num = trained_bot.select_move(board)
            time.sleep(1)
        else:
            col_num = int(input("pick a col 0-6: "))

        board.drop_piece(col_num)
        print(board)

    print(f'Player {Player(board.winner)} is the winner!')



def watch_bots_play():
    board = Connect4Board()

    A = np.array(json.loads(open('/Users/darrendeanto/workspace/connect4/A.json').read()))

    untrained_bot = MatrixBot(player=Player.X)
    trained_bot = MatrixBot(player=Player.O, A=A)

    while not board.winner:
        if board.whose_turn == untrained_bot.player:
            col_num = untrained_bot.select_move(board)
            time.sleep(0.5)
        else:
            col_num = trained_bot.select_move(board)
            time.sleep(0.5)


        board.drop_piece(col_num)
        print(board)

    print(f'Player {Player(board.winner)} is the winner!')



def repeated_bots_play_fast(num_rounds: int=200):
    from scipy.stats import binom

    A = np.array(json.loads(open('/Users/darrendeanto/workspace/connect4/A.json').read()))

    wins = [0, 0, 0]
    for round in range(num_rounds):
        board = Connect4Board()

        untrained_bot = MatrixBot(player=Player.X)
        trained_bot = MatrixBot(player=Player.O, A=A)

        while not board.winner:
            if board.whose_turn == trained_bot.player:
                col_num = trained_bot.select_move(board)
            else:
                col_num = untrained_bot.select_move(board)

            board.drop_piece(col_num)

        if Player(board.winner) == trained_bot.player:
            wins[0] += 1
        elif Player(board.winner) == untrained_bot.player:
            wins[1] += 1
        else:
            wins[2] += 1

        print(wins, Player(board.winner))


if __name__ == '__main__':
    repeated_bots_play_fast()