import random
from game_board import Connect4Board, Player
import copy
import time


"""
is good move if no children result in loss and each of those children have >1 good move for me
"""

count = 0


def _count(depth: int):
    global count

    count += 1
    if count % 10_000 == 0:
        print(f'depth: {depth}, count: {count}') 
 

class Bot():
    def __init__(self, max_depth: int):
        assert max_depth >= 0

        self.me = Player.O
        self.enemy = Player.X
        self.max_depth = max_depth


    def _is_good_move(self, parent: Connect4Board, depth: int = 0) -> bool:
        if depth >= self.max_depth:
            return True

        if parent.winner == self.me:
            return True

        # if any of their moves result in loss then is not good move
        for their_move in parent.make_children():
            if their_move.winner == self.enemy:
                return False

            atleast_1_good_move = False
            for my_move in their_move.make_children():
                if self._is_good_move(my_move, depth+2):
                    atleast_1_good_move = True
                    break
            
            if not atleast_1_good_move:
                return False

        return True


    def select_move(self, board: Connect4Board) -> int:
        assert board.whose_turn == self.me

        possible_moves = list(board.make_children())
        random.shuffle(possible_moves)

        for move in possible_moves:
            if self._is_good_move(parent=move):
                # found first good move, return col_num that depicts this move
                return move.last_move[1]

        # if no good moves, pick bad move at random cuz f it 
        return random.choice(possible_moves).last_move[1]
