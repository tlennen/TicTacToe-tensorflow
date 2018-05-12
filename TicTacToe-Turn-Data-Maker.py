import random
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# This program creates the data sets for the prediction models of Tic-Tac-Toe. The game is player between
# an AI and a random selector. Every time a move is made, the board state is saved to an array. Once the match
# is won, the winner is appended to every board state in the array. The data in the array then saved to an
# Excel file for use in the prediction model.

# In this version of the program, the program also keeps track of who last played. This allows the network to
# have more information and be more accurate in its predictions.

# There are several ways that the data is preprocessed before it is sent to the Excel file. The board states
# are first converted into numbers (0 for none, 1 for 'X', 2 for 'O'). The board is then rotated 3 times around
# the grid. Because Tic-Tac-Toe is uniform and orientation does not matter, this is basically like playing
# 3 additional games at the same time. The data is then doubled by reversing the 1s and 2s so there is more
# variation in the data set. The Excel file name must not exist before running the program.

# The game is played an X number of times (can be changed down below) and then saved to Excel file.
# The Excel file must be changed to look like Example.csv and saved as an csv. The first row is the following
# values respectively: # of data sets, # of columns, and the possible outcomes. # of data sets can be found
# by scrolling to the bottom and subtracting the row number by 1.

# Created by Tyler Lennen
# Last updated May 11th, 2018

number_of_games = 500

class Tic(object):
    winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6])

    winners = ('X-win', 'Draw', 'O-win')

    def __init__(self, squares=[]):
        if len(squares) == 0:
            self.squares = [None for i in range(9)]
        else:
            self.squares = squares

    def show(self):
        for element in [self.squares[i:i + 3] for i in range(0, len(self.squares), 3)]:
            print(element)

    def available_moves(self):
        """what spots are left empty?"""
        return [k for k, v in enumerate(self.squares) if v is None]

    def available_combos(self, player):
        """what combos are available?"""
        return self.available_moves() + self.get_squares(player)

    def complete(self):
        """is the game over?"""
        if None not in [v for v in self.squares]:
            return True
        if self.winner() != None:
            return True
        return False

    def X_won(self):
        return self.winner() == 'X'

    def O_won(self):
        return self.winner() == 'O'

    def tied(self):
        return self.complete() == True and self.winner() is None

    def winner(self):
        for player in ('X', 'O'):
            positions = self.get_squares(player)
            for combo in self.winning_combos:
                win = True
                for pos in combo:
                    if pos not in positions:
                        win = False
                if win:
                    return player
        return None

    def get_squares(self, player):
        """squares that belong to a player"""
        return [k for k, v in enumerate(self.squares) if v == player]

    def make_move(self, position, player):
        """place on square on the board"""
        self.squares[position] = player

    def alphabeta(self, node, player, alpha, beta):
        if node.complete():
            if node.X_won():
                return -10
            elif node.tied():
                return 0
            elif node.O_won():
                return 10
        for move in node.available_moves():
            node.make_move(move, player)
            val = self.alphabeta(node, get_enemy(player), alpha, beta)
            node.make_move(move, None)
            if player == 'O':
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    return beta
            else:
                if val < beta:
                    beta = val
                if beta <= alpha:
                    return alpha
        if player == 'O':
            return alpha
        else:
            return beta

    def get_square(self):
        return self.squares

    def get_winner(self):
        return self.winner()


def determine(board, player):
    a = -2
    choices = []
    if len(board.available_moves()) == 9:
        return 4
    for move in board.available_moves():
        board.make_move(move, player)
        val = board.alphabeta(board, get_enemy(player), -2, 2)
        board.make_move(move, None)
        if val > a:
            a = val
            choices = [move]
        elif val == a:
            choices.append(move)
    return random.choice(choices)


def get_enemy(player):
    if player == 'X':
        return 'O'
    return 'X'


def convert_board(board,turn):
    hold = [];
    for x in board:
        if x == 'X':
            hold.append(1)
        elif x == 'O':
            hold.append(2)
        else:
            hold.append(0)
    hold.append(turn)
    return hold


def rotate_answers(hold):
    size = len(hold)
    for x in range(0,size*3):
        temp=[0,0,0,0,0,0,0,0,0,0,0]
        temp[0] = hold[x][6]
        temp[1] = hold[x][3]
        temp[2] = hold[x][0]
        temp[3] = hold[x][7]
        temp[5] = hold[x][1]
        temp[6] = hold[x][8]
        temp[7] = hold[x][5]
        temp[8] = hold[x][2]
        temp[4] = hold[x][4]
        temp[9] = hold[x][9]
        temp[10]= hold[x][10]
        hold.append(temp)
    return hold


def double_hold(hold):
    size = len(hold)
    for x in range(0, size):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for y in range(0,len(hold[x])):
            if hold[x][y] == 2:
                temp[y] = 1
            elif hold[x][y] == 1:
                temp[y] = 2
        hold.append(temp)
    return hold


def one_game():
    board = Tic()
    hold = []
    while not board.complete():
        player = 'X'
        player_move = random.randrange(0,9)
        if not player_move in board.available_moves():
            continue
        board.make_move(player_move, player)
        hold.append(convert_board(board.get_square(),1))
        if board.complete():
            break
        player = get_enemy(player)
        computer_move = determine(board, player)
        board.make_move(computer_move, player)
        hold.append(convert_board(board.get_square(),2))
    winner = board.winner()
    if winner == 'X':
        winner = 1
    elif winner == 'O':
        winner = 2
    else:
        winner = 0
    print(hold)
    for x in hold:
        x.append(winner)
    hold = rotate_answers(hold)
    double_hold(hold)
    print("winner is", board.winner())
    return hold


if __name__ == "__main__":
    num_rows = 0
    hold = []
    for x in range(0,number_of_games):
        temp = (one_game())
        for y in temp:
            hold.append(y)
    df = pd.DataFrame(hold)
    writer = pd.ExcelWriter('TurnTestingData.xlsx', engine='openpyxl')
    df.to_excel(writer, startrow=0, index=False)
    writer.save()