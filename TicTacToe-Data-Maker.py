import random
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

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
            print (element)

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
        #print ("move:", move + 1, "causes:", board.winners[val]) ##Displays the score of each move
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

def convert_board(board):
    hold = [];
    for x in board:
        if x == 'X':
            hold.append(1)
        elif x == 'O':
            hold.append(2)
        else:
            hold.append(0)
    return hold

def rotate_answers(hold):
    size = len(hold)
    print("FIRST ",hold[0][0])
    for x in range(0,size*3):
        temp=[0,0,0,0,0,0,0,0,0,0]
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
        hold.append(temp)
    return hold;

def double_hold(hold):
    size = len(hold)
    for x in range(0,size):
        temp = [0,0,0,0,0,0,0,0,0,0]
        temp = hold[x]
        for y in range(0,len(temp)):
            if temp[y] == 1:
                temp[y] = 2
            elif temp[y] == 2:
                temp[y]==1
        hold.append(temp)
    return hold

def one_game():
    board = Tic()
    board.show()
    hold = []
    while not board.complete():
        player = 'X'
        player_move = random.randrange(0,9)
        #print("PLAYER MOVE: ",player_move)
        #board.show()
        if not player_move in board.available_moves():
            continue
        board.make_move(player_move, player)
        board.show()
        hold.append(convert_board(board.get_square()))
        if board.complete():
            break
        player = get_enemy(player)
        computer_move = determine(board, player)
        board.make_move(computer_move, player)
        board.show()
        hold.append(convert_board(board.get_square()))
    winner = board.winner()
    print(winner)
    if winner == 'X':
        winner = 1;
    elif winner == 'O':
        winner = 2;
    else:
        winner = 0;
    print(winner)
    for x in hold:
        x.append(winner)
    print(hold)
    hold = rotate_answers(hold)
    print(hold)
    #double_hold(hold);
    print(hold)
    print ("winner is", board.winner())
    return hold

if __name__ == "__main__":
    num_rows = 0
    hold = []
    for x in range(0,1):
        temp = (one_game())
        for y in temp:
            hold.append(y)
    """df = pd.DataFrame(hold)
    writer = pd.ExcelWriter('test.xlsx', engine='openpyxl')
    df.to_excel(writer, startrow=0, index=False)
    writer.save()"""