import cv2
from detect import get_board
from engine import getBestMove

move_code = {
    'U': (-1, 0),
    'D': (1, 0),
    'R': (0, 1),
    'L': (0, -1)
}

columns = 7
rows = 5
img = cv2.imread('Screenshot_20211203-233841.png')
board, coord_board = get_board(img, rows, columns)

if board:
    for b in board:
        print(b)
    print()
    best_move = getBestMove(board)
    x = int(best_move[3])
    y = int(best_move[1])
    move = best_move[-1]
    toLoc = move_code[best_move[-1]]
    print(x, y, move)
    print(coord_board[x][y])
    print(coord_board[x + toLoc[0]][y + toLoc[1]])

