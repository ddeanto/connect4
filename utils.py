import numpy as np

ROWS = 6
COLS = 7

def _generate_all_Nsegments(n: int):
    """
    Generate all N(=4)segments on the board.
    """
    segments = []

    # horizontal segments
    for col in range(COLS - (n-1)):
        for row in range(ROWS):
            segments.append([(row, col + i) for i in range(n)])

    # vertical segments
    for col in range(COLS):
        for row in range(ROWS - (n-1)):
            segments.append([(row + i, col) for i in range(n)])

    # negatively sloped diagonals
    for col in range(COLS - (n-1)):
        for row in range(ROWS - (n-1)):
            segments.append([(row + i, col + i) for i in range(n)])

    # positively sloped diagonals
    for col in range(COLS - (n-1)):
        for row in range(n-1, ROWS):
            segments.append([(row - i, col + i) for i in range(n)])

    return segments


def generate_cell_to_segments_map():
    """
    For each cell on the game board, find the segments that touch it, and put in dict.
    """
    segments = _generate_all_Nsegments(n=4)

    map = {}
    for col in range(COLS):
        for row in range(ROWS):
            matches = [s for s in segments if (row, col) in s]
            map[(row, col)] = matches
    return map


CELL_TO_SEGMENTS_MAP = generate_cell_to_segments_map()


def generate_all_segments():
    segments = []
    for n in range(1, max(ROWS, COLS)):
        segments.extend(_generate_all_Nsegments(n=n))

    for n in range(len(segments)):
        segments[n] = np.array(segments[n])

    return segments
