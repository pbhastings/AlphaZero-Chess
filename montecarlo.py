# https://www.youtube.com/watch?v=UXW2yZndl7U

from cmath import inf
from math import sqrt
import math
import chess
import evaluator

class Node:
    c = 1 # c always >=0, high c means more exploring low visited nodes

    def __init__(self, board: chess.Board):
        self.children = []
        self.board = board
        self.n = 0
        self.w = 0 #w incremented when white wins the monte carlo in child, so sometimes may need to flip sign for black
    
    def calc_uct(self, np):
        # this is based on an algorithm from multi-armed bandits problem
        win_prob = self.w / self.n
        if (not self.board.turn): # look at moves that are good for black if black's turn?
            win_prob *= -1
        explore = Node.c * sqrt(math.log(np) / self.n)
        return win_prob + explore
        

    def add_children(self):
        # here we can return (n, w) tuple to pass back up and add along the way
        #n, w = 0
        for move in self.board.legal_moves:
            new_board = self.board.copy(stack=False)
            new_board.push(move)
            #ev = evaluator.mc_eval(new_board)
            #w += ev
            #n += 1
            self.children.append(Node(new_board))
        #return (n, w)

    def rollout(self):
        ev = evaluator.mc_eval(self.board)
        self.w = ev
        self.n += 1
        return ev
    
    def get_maximizing_child(self):
        m = -math.inf
        maximizing_child = None
        for child in self.children:
            if (child.n == 0):
                return child
            child_uct = child.calc_uct(self.n)
            if (child_uct > m):
                m = child_uct
                maximizing_child = child
        return maximizing_child

    def mcts(self):
        if (len(self.children) == 0):
            if (self.n == 0):
                return self.rollout()
            else:
                self.add_children()
                return self.children[0].rollout()
        else:
            w = self.get_maximizing_child().mcts()
            self.n += 1
            self.w += w
            return w
    
    def get_best_move(self, num):
        for i in range(num):
            self.mcts()
        most_visited = -1
        next_pos = None
        for child in self.children:
            if child.n > most_visited:
                most_visited = child.n
                next_pos = child.board
        return next_pos



    def get_num_children(self):
        if (len(self.children) == 0):
            return 1
        
        n = 0
        for child in self.children:
            n += child.get_num_children()
        return n