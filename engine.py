# Created by MrBBS
# 12/5/2021
# -*-encoding:utf-8-*-

import copy
import pandas as pd


def verticalSearch(board):
    moves = []
    range_h = len(board) - 1
    range_w = len(board[0]) - 1
    for x in range(range_w + 1):
        curColor = ""
        for y in range(range_h + 1):
            if board[y][x] != curColor:
                curColor = board[y][x]
                if y + 2 < range_h and board[y + 2][x] == curColor:
                    if x - 1 >= 0 and board[y + 1][x - 1] == curColor:
                        moves.append("({},{}){}".format(x, y + 1, "L"))
                    if x + 1 < range_w and board[y + 1][x + 1] == curColor:
                        moves.append("({},{}){}".format(x, y + 1, "R"))
            else:
                if y - 3 >= 0 and board[y - 3][x] == curColor:
                    moves.append("({},{}){}".format(x, y - 2, "U"))
                if y + 2 < range_h and board[y + 2][x] == curColor:
                    moves.append("({},{}){}".format(x, y + 1, "D"))
                if y - 2 >= 0 and x - 1 >= 0 and board[y - 2][x - 1] == curColor:
                    moves.append("({},{}){}".format(x, y - 2, "L"))
                if y - 2 >= 0 and x + 1 < 8 and board[y - 2][x + 1] == curColor:
                    moves.append("({},{}){}".format(x, y - 2, "R"))
                if y + 1 < range_h and x - 1 >= 0 and board[y + 1][x - 1] == curColor:
                    moves.append("({},{}){}".format(x, y + 1, "L"))
                if y + 1 < range_h and x + 1 < range_w and board[y + 1][x + 1] == curColor:
                    moves.append("({},{}){}".format(x, y + 1, "R"))
                curColor = ""
    return moves


def horizontalSearch(board):
    moves = []
    range_h = len(board) - 1
    range_w = len(board[0]) - 1
    for y in range(range_h + 1):
        curColor = ""
        for x in range(range_w + 1):
            if board[y][x] != curColor:
                curColor = board[y][x]
                if x + 2 < range_w and board[y][x + 2] == curColor:
                    if y - 1 >= 0 and board[y - 1][x + 1] == curColor:
                        moves.append("({},{}){}".format(x + 1, y, "U"))
                    if y + 1 < range_h and board[y + 1][x + 1] == curColor:
                        moves.append("({},{}){}".format(x + 1, y, "D"))
            else:
                if x - 3 >= 0 and board[y][x - 3] == curColor:
                    moves.append("({},{}){}".format(x - 2, y, "L"))
                if x + 2 < range_w and board[y][x + 2] == curColor:
                    moves.append("({},{}){}".format(x + 1, y, "R"))
                if y - 1 >= 0 and x - 2 >= 0 and board[y - 1][x - 2] == curColor:
                    moves.append("({},{}){}".format(x - 2, y, "U"))
                if y - 1 >= 0 and x + 1 < range_w and board[y - 1][x + 1] == curColor:
                    moves.append("({},{}){}".format(x + 1, y, "U"))
                if y + 1 < range_h and x - 2 >= 0 and board[y + 1][x - 2] == curColor:
                    moves.append("({},{}){}".format(x - 2, y, "D"))
                if y + 1 < range_h and x + 1 < range_w and board[y + 1][x + 1] == curColor:
                    moves.append("({},{}){}".format(x + 1, y, "D"))
                curColor = ""
    return moves


def processBoard(board):
    moves = []
    for move in verticalSearch(board):
        moves.append(move)
    for move in horizontalSearch(board):
        moves.append(move)
    return moves


def _getMatchType(board, x, y, d):
    range_h = len(board) - 1
    range_w = len(board[0]) - 1
    ways = {1: 0, 2: 0, 3: 0}
    print("Getting move for ({}, {})".format(x, y))
    # Get the number of a color gem in a consecutive line
    if d == "U":
        c = board[y - 1][x]
        for i in range(1, 3):
            # Down
            if y + i < range_h and board[y + i][x] == c:
                if i == 2 and ways[2] == 0:
                    continue
                ways[2] = i
            # Right
            if x + i < range_w and board[y][x + i] == c:
                if i == 2 and ways[3] == 0:
                    continue
                ways[3] = i
            # Left
            if x - i >= 0 and board[y][x - i] == c:
                if i == 2 and ways[1] == 0:
                    continue
                ways[1] = i
    elif d == "D":
        c = board[y + 1][x]
        for i in range(1, 3):
            # Up
            if y - i >= 0 and board[y - i][x] == c:
                if i == 2 and ways[2] == 0:
                    continue
                ways[2] = i
            # Right
            if x + i < range_w and board[y][x + i] == c:
                if i == 2 and ways[3] == 0:
                    continue
                ways[3] = i
            # Left
            if x - i >= 0 and board[y][x - i] == c:
                if i == 2 and ways[1] == 0:
                    continue
                ways[1] = i
    elif d == "L":
        c = board[y][x - 1]
        for i in range(1, 3):
            # Down
            if y + i < range_h and board[y + i][x] == c:
                if i == 2 and ways[1] == 0:
                    continue
                ways[1] = i
            # Up
            if y - i >= 0 and board[y - i][x] == c:
                if i == 2 and ways[3] == 0:
                    continue
                ways[3] = i
            # Right
            if x + i < range_w and board[y][x + i] == c:
                if i == 2 and ways[2] == 0:
                    continue
                ways[2] = i
    else:
        c = board[y][x + 1]
        for i in range(1, 3):
            # Down
            if y + i < range_h and board[y + i][x] == c:
                if i == 2 and ways[1] == 0:
                    continue
                ways[1] = i
            # Up
            if y - i >= 0 and board[y - i][x] == c:
                if i == 2 and ways[3] == 0:
                    continue
                ways[3] = i
            # Left
            if x - i >= 0 and board[y][x - i] == c:
                if i == 2 and ways[2] == 0:
                    continue
                ways[2] = i
    if ways[1] == 2 and ways[2] == 2 and ways[3] == 2:
        theType = "3W"
    elif ways[1] > 0 and ways[3] > 0 and ways[2] == 2:
        theType = "T"
    elif (ways[1] == 2 or ways[3] == 2) and ways[2] == 2:
        theType = "L"
    elif ways[1] == 2 and ways[3] == 2:
        theType = "5M"
    elif (ways[1] == 2 and ways[3] == 1) or (ways[1] == 1 and ways[3] == 2):
        theType = "4M"
    else:
        theType = "3M"
    return theType


def checkDoubleMove(moves):
    doubles = []
    for move in moves:
        x = int(move[1])
        y = int(move[3])
        d = move[-1]
        if d == "R":
            x += 1
            d = "L"
        elif d == "L":
            x -= 1
            d = "R"
        elif d == "U":
            y -= 1
            d = "D"
        else:
            y += 1
            d = "U"
        newPos = "({},{}){}".format(x, y, d)
        if newPos in moves:
            doubles.append(move)
    return doubles


def predictAfterMove(move, board):
    d = move[-1]
    x = int(move[1])
    y = int(move[3])
    range_h = len(board) - 1
    range_w = len(board[0]) - 1
    spots = []
    if d == "R":
        tmp = board[y][x + 1]
        board[y][x + 1] = board[y][x]
        board[y][x] = tmp
    elif d == "L":
        tmp = board[y][x - 1]
        board[y][x - 1] = board[y][x]
        board[y][x] = tmp

    elif d == "U":
        tmp = board[y - 1][x]
        board[y - 1][x] = board[y][x]
        board[y][x] = tmp
    else:
        tmp = board[y + 1][x]
        board[y + 1][x] = board[y][x]
        board[y][x] = tmp
    # Get the spots that are now a match
    # Vertical search
    for xCur in range(range_w + 1):
        for yCur in range(range_h - 1):
            if board[yCur][xCur] == board[yCur + 1][xCur] and \
                    board[yCur + 1][xCur] == board[yCur + 2][xCur]:
                if (yCur, xCur) not in spots:
                    spots.append((yCur, xCur))
                if (yCur + 1, xCur) not in spots:
                    spots.append((yCur + 1, xCur))
                if (yCur + 2, xCur) not in spots:
                    spots.append((yCur + 2, xCur))
    # Horizontal search
    for yCur in range(range_h + 1):
        for xCur in range(range_w - 1):
            if board[yCur][xCur] == board[yCur][xCur + 1] and \
                    board[yCur][xCur + 1] == board[yCur][xCur + 2]:
                if (yCur, xCur) not in spots:
                    spots.append((yCur, xCur))
                if (yCur, xCur + 1) not in spots:
                    spots.append((yCur, xCur + 1))
                if (yCur, xCur + 2) not in spots:
                    spots.append((yCur, xCur + 2))
    # Now that we have the matches, predict after effect.
    spots.sort()
    for curY, curX in spots:
        while curY != 0:
            board[curY][curX] = board[curY - 1][curX]
            curY -= 1
        board[0][curX] = 'N/A'
    return board


def getBestMove(board):
    moves = processBoard(board)
    doubles = checkDoubleMove(moves)
    range_h = len(board) - 1
    range_w = len(board[0]) - 1
    moveData = {"move": [], "type": [], "double": [], "reaction": []}
    for move in moves:
        theType = _getMatchType(
            board,
            int(move[1]),
            int(move[3]),
            move[-1]
        )
        isDouble = move in doubles
        afterBoard = predictAfterMove(move, copy.deepcopy(board))
        #
        cascadeAmount = 0
        for x in range(range_w + 1):
            for y in range(range_h - 1):
                if afterBoard[y][x] == 'N/A':
                    continue
                if afterBoard[y][x] == afterBoard[y + 1][x] and afterBoard[y + 1][x] == afterBoard[y + 2][x]:
                    cascadeAmount += 1
        for y in range(range_h + 1):
            for x in range(range_w - 1):
                if afterBoard[y][x] == 'N/A':
                    continue
                if afterBoard[y][x] == afterBoard[y][x + 1] and afterBoard[y][x + 1] == afterBoard[y][x + 2]:
                    cascadeAmount += 1
        moveData['move'].append(move)
        moveData['type'].append(theType)
        moveData['double'].append(isDouble)
        moveData['reaction'].append(cascadeAmount)
    matchOrdering = ["3W", "T", "L", "5M", "4M", "3M"]
    moveInfo = pd.DataFrame(moveData, columns=["move", "type", "double", "reaction"])

    moveInfo.type = moveInfo.type.astype("category")
    moveInfo.type.cat.set_categories(matchOrdering, inplace=True)

    moveInfo.sort_values(["type", "double", "reaction"], ascending=[True, False, False], inplace=True)
    print(moveInfo.head(len(moveInfo)))
    bestMove = moveInfo.head(1).reset_index(drop=True)["move"][0]

    return bestMove
