import numpy as np
from enum import Enum
import copy

from utils import CELL_TO_SEGMENTS_MAP


class Player(int, Enum):
    X = -1
    O = 1
    EMPTY = 0


class ColumnFullError(ValueError):
    pass


class Connect4Board():
    def __init__(self):
        self.board = np.array([[Player.EMPTY]*7]*6)
        self.winner = None
        self.whose_turn = Player.X
        self.last_move = None

        self.cells_open = 7*6
        self.column_depth = [6-1]*7

    
    def __str__(self):
        output_str = ''

        rows = [self.board[n, :] for n in range(6)]
        for row in rows:
            for cell in row:
                output_str += ' ' + Player(cell).name
            output_str += '\n'

        s = [str(n) for n in range(7)]
        output_str += f'\n {" ".join(s)}\n'

        return output_str.replace('EMPTY', '-')

    
    def _toggle_turn(self):
        self.whose_turn *= -1

    
    def _put_piece_in_slot(self, row_num: int, col_num: int):
        self.board[row_num, col_num] = self.whose_turn
        self.last_move = (row_num, col_num)


    def make_children(self):
        for col_num in range(7):
            if self.column_depth[col_num] >= 0:
                child = copy.deepcopy(self)
                child.drop_piece(col_num=col_num)
                yield child

    
    def drop_piece(self, col_num: int):
        row = self.column_depth[col_num]
        if row == -1:
            raise ColumnFullError

        self.column_depth[col_num] -= 1
        
        # then let's just add it to the actual board
        self._put_piece_in_slot(row_num=row, col_num=col_num)

        self.cells_open -= 1

        self.who_won()

        # toggle turn
        self._toggle_turn()

    
    def who_won(self):
        """
        Check who won by:
         - using map to get relevant N(=4)segments
         - check any segments are N(=4) X's or Os in a row.
        """
        if self.cells_open <= 0:
            return Player.EMPTY

        # 4 X's or O's in a row. To be compared to.
        win = [self.whose_turn] * 4

        # get all 4segments that contain last_move's cell
        segments = CELL_TO_SEGMENTS_MAP[self.last_move]

        # iterate over each segment and see if they are 4 X's or O's in a row
        for segment in segments:
            cell_values = [self.board[r, c] for r, c in segment]
            if cell_values == win:
                self.winner = self.whose_turn
                break
