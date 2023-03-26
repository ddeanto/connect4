from bot import Bot
from game_board import Connect4Board, Player
import time


if __name__ == '__main__':
    board = Connect4Board()
    bot = Bot(depth=6)

    while not board.winner:
        if board.whose_turn == Player.X:
            col_num = int(
                input(f'Player {board.whose_turn}, Make your Selection(0-6):'))            
        else:
            begin = time.time()

            col_num = bot.select_move(board)

            duration = time.time() - begin
            if time.time() - begin < 1:
                time.sleep(0.5)


        board.drop_piece(col_num)
        print(board)

    print(f'Player {Player(board.winner)} is the winner!')