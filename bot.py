import random
from game_board import Connect4Board, Player
import copy
import time


"""
is good move if no children result in loss and each of those children have >1 good move for me
"""

def _count(depth: int):
    global count

    count += 1
    if count % 10_000 == 0:
        print(f'depth: {depth}, count: {count}') 
 

class Bot():
    def __init__(self, depth: int):
        self.me = Player.O
        self.enemy = Player.X
        self.depth = depth


    def select_move(self, board: Connect4Board):
        assert board.whose_turn == self.me

        possible_moves = list(board.make_children())
        random.shuffle(possible_moves)

        for move in possible_moves:
            if self.is_good_move(move, self.depth):
                return move.last_move[1]

        return random.choice(possible_moves).last_move[1]

        

    def is_good_move(self, parent: Connect4Board, max_depth: int, depth: int = 0) -> bool:
        if depth > max_depth:
            return True

        if parent.winner == self.me:
            return True

        # generate their moves
        their_moves = parent.make_children()

        # if any of their moves result in loss then is not good move
        for their_move in their_moves:
            if their_move.winner == self.enemy:
                return False

            # generate my moves
            my_moves = their_move.make_children()

            atleast_1_good_move = False
            for my_move in my_moves:  # TODO: make generator
                if self.is_good_move(my_move, max_depth, depth+2):
                    atleast_1_good_move = True
                    break
            
            if not atleast_1_good_move:
                return False

        return True


count = 0