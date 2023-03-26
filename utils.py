
def _generate_all_4segments():
    """
    Generate all N(=4)segments on the board.
    """
    segments = []

    # horizontal segments
    for col in range(7 - 3):
        for row in range(6):
            segments.append([(row, col + i) for i in range(4)])

    # vertical segments
    for col in range(7):
        for row in range(6 - 3):
            segments.append([(row + i, col) for i in range(4)])

    # negatively sloped diagonals
    for col in range(7 - 3):
        for row in range(6 - 3):
            segments.append([(row + i, col + i) for i in range(4)])

    # positively sloped diagonals
    for col in range(7 - 3):
        for row in range(3, 6):
            segments.append([(row - i, col + i) for i in range(4)])

    return segments


def generate_cell_to_segments_map():
    """
    For each cell on the game board, find the segments that touch it, and put in dict.
    """
    segments = _generate_all_4segments()

    map = {}
    for col_num in range(7):
        for row_num in range(6):
            matches = [s for s in segments if (row_num, col_num) in s]
            map[(row_num, col_num)] = matches
    return map


CELL_TO_SEGMENTS_MAP = generate_cell_to_segments_map()
