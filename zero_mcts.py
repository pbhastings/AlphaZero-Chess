from cmath import inf
from math import sqrt
import math

import constants
import chess
import chess.pgn
import btm
import zeromodel

import pandas as pd
import sys
import numpy as np

class ZeroEdge:
    def __init__(self, parent, child, move: chess.Move, p):
        self.parent = parent
        self.child = child
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.move = move #move that goes from parent position to child position

class ZeroNode:
    def __init__(self, board: chess.Board):
        self.edges = [] #edges leading to children
        self.board = board
    
    def is_leaf(self):
        if (len(self.edges) == 0):
            return True
        return False

    def add_edge(self, edge: ZeroEdge):
        self.edges.append(edge)

    def num_children(self):
        if (len(self.edges) == 0):
            return 1
        
        n = 0
        for edge in self.edges:
            child = edge.child
            n += child.num_children()
        return n

class MCTS:
    def __init__(self, root: ZeroNode):
        self.root = root
        #self.nodes = {} #dictionary containing all nodes for lookup purposes; two nodes
    
    #selects the next leaf node to evaluate based on alphazero's mcts algorithm,
    #returns that node and the path from the root to the node
    def select(self):
        cur_node = self.root
        path = []

        while not cur_node.is_leaf():
            max_qu = -inf
            nb = 0
            maximizing_edge = None

            if (cur_node == self.root):
                dirichlet = np.random.dirichlet([constants.DIRICHLET_ALPHA] * len(cur_node.edges))

            for edge in cur_node.edges:
                nb += edge.n
            
            for i, edge in enumerate(cur_node.edges):
                q = edge.q
                p = edge.p
                if (cur_node == self.root):
                    p = constants.DIRICHLET_EPSILON * dirichlet[i] + (1 - constants.DIRICHLET_EPSILON) * p
                u = constants.CPUCT * p * sqrt(nb) / (1 + edge.n)
                if q+u > max_qu:
                    max_qu = q+u
                    maximizing_edge = edge
            
            path.append(maximizing_edge)
            cur_node = maximizing_edge.child
        
        return cur_node, path
    
    #given the path mcts took to reach a leaf and the value of that leaf (predicted or value based on game result), update edge values along the path:
    #increment n, add value to w (based on whose turn it is and whose turn the leaf board is), update q
    def backprop(self, path, value, leaf: ZeroNode):
        leaf_color = leaf.board.turn

        for edge in path:
            parent_color = edge.parent.board.turn #color of player who played the move that is this edge
            if parent_color != leaf_color: #e.g. if the leaf player is black and the value of that board is 0.75 for black, we should subtract the value from edges where white plays into the line
                turn_parity = -1
            else:
                turn_parity = 1
            
            edge.n += 1
            edge.w += value * turn_parity
            edge.q = edge.w / edge.n
    
    def print_tree(self):
        print(self.root.board)
        for e in self.root.edges:
            print(e.move, e.n)