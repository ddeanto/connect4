from neural_net import NeuralNet
from game_board import Connect4Board, Player
from bot import Bot


if __name__ == '__main__':
    trained_playerX = NeuralNet.from_json('player-X.json')
    tree_bot = Bot(max_depth=2)

    wins = [0, 0]
    for _ in range(1_000):
        board = Connect4Board()

        while board.winner is None:
            if board.whose_turn == Player.X:
                col_num = trained_playerX.select_move(board)
            else:
                col_num = tree_bot.select_move(board)

            board.drop_piece(col_num)

        winner = Player(board.winner)
        if winner == Player.X:
            wins[0] += 1
        elif winner == Player.O:
            wins[1] += 1

        print(f'\nwinner:{winner}\n{board}')
        print(wins)
        # time.sleep(0.0)

