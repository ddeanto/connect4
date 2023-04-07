import sys
import time

from main import Connect4Board, Player
from neural_net import NeuralNet


def test_4segments_generation():
    """
    In a RxC board of connectN there should be
    (R+1-N)*C verticals
    R*(C+1-N) horizontals
    (R+1-N)*(C+1-N) positive diagonals
    (R+1-N)*(C+1-N) negative diagonals
    ...
    num of Nsegments = (R+1-N)*C + R*(C+1-N) + 2*(R+1-N)*(C+1-N)
    """
    board4 = Connect4Board()

    print(f'size of board empty: {sys.getsizeof(board4)}')
    board4.drop_piece(0)
    board4.drop_piece(0)
    board4.drop_piece(0)
    board4.drop_piece(0)
    board4.drop_piece(1)
    board4.drop_piece(1)
    board4.drop_piece(1)
    board4.drop_piece(1)
    print(f'size of partially filled: {sys.getsizeof(board4)}')

    all_4segments = board4._generate_all_4segments()

    R = 6
    C = 7
    N = 4
    num_4segments_expected = (R+1-N)*C + R*(C+1-N) + 2*(R+1-N)*(C+1-N)

    assert num_4segments_expected == len(all_4segments)


def test_cell_to_segments_map1():
    board4 = Connect4Board()
    segments = board4.cell_to_segments_map[(0,0)]

    expected_segments = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 1), (2, 2), (3, 3)],
    ]
    
    for segment in expected_segments:
        assert segment in segments

    assert len(expected_segments) == len(segments)


def test_cell_to_segments_map2():
    board4 = Connect4Board()
    segments = board4.cell_to_segments_map[(2,2)]

    expected_segments = [
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(2, 1), (2, 2), (2, 3), (2, 4)],
        [(2, 2), (2, 3), (2, 4), (2, 5)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 2), (2, 2), (3, 2), (4, 2)],
        [(2, 2), (3, 2), (4, 2), (5, 2)],
        [(0, 0), (1, 1), (2, 2), (3, 3)],
        [(1, 1), (2, 2), (3, 3), (4, 4)],
        [(2, 2), (3, 3), (4, 4), (5, 5)],
        [(4, 0), (3, 1), (2, 2), (1, 3)],
        [(3, 1), (2, 2), (1, 3), (0, 4)]
    ]
    
    for segment in expected_segments:
        assert segment in segments

    assert len(expected_segments) == len(segments)


def test_make_move():
    board4 = Connect4Board()
    
    board4.drop_piece(3)
    assert board4.board[5, 3] == 1
    
    board4.drop_piece(3)
    assert board4.board[4, 3] == -1
    

def test_who_won1():
    board4 = Connect4Board()
    
    board4.drop_piece(2)
    board4.drop_piece(1)
    board4.drop_piece(2)
    board4.drop_piece(3)
    board4.drop_piece(2)
    board4.drop_piece(5)
    board4.who_won()
    print(f'line 84 no1 won yet\n{board4}')
    assert not board4.winner
    
    board4.drop_piece(2)
    board4.who_won()
    print(f'line 89 o won\n{board4}')
    assert board4.winner == 1


def test_who_won2():
    board4 = Connect4Board()

    board4.drop_piece(0)
    board4.drop_piece(2)
    board4.drop_piece(1)
    board4.drop_piece(2)
    board4.drop_piece(3)
    board4.drop_piece(2)
    board4.drop_piece(5)
    board4.who_won()
    print(f'line 104 no1 won yet\n{board4}')
    assert not board4.winner
    
    board4.drop_piece(2)
    board4.who_won()
    print(f'line 109 x won\n{board4}')
    assert board4.winner == -1


def test_tie():
    board4 = Connect4Board()
    for _ in range(2):
        for col in range(7):
            board4.drop_piece(col)
            board4.drop_piece((col+1)%7)
            board4.drop_piece((col+2)%7)
    
    assert board4.winner == Player.EMPTY


def test_benchmark():
    begin = time.time()

    for _ in range(30*30):
        player_x = NeuralNet()
        player_o = NeuralNet()

        board = Connect4Board()
        
        while board.winner is None:
            if board.whose_turn == Player.X:
                col = player_x.select_move(board=board)
            else:
                col = player_o.select_move(board=board)
            board.drop_piece(col)

    end = time.time()
    print(round(end-begin, 2))

