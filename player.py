from cmath import inf
from math import sqrt
import math
import random

import constants
import chess
import chess.pgn
import btm
import zeromodel
import zero_mcts
import board_eval

import numpy as np
import tensorflow as tf
from keras.models import Model

class Player:
    #assigns root_node based on the given board, ands inits mcts and model
    def __init__(self, board: chess.Board, model: Model):
        root_node = zero_mcts.ZeroNode(board)
        self.mcts = zero_mcts.MCTS(root_node)
        self.model = model
    
    def run_mcts(self, simulations):
        for i in range(simulations):
            selected_node, path = self.mcts.select()
            value = self.expand_and_evaluate(selected_node) #selected_node is a leaf because of how mcts works
            self.mcts.backprop(path, value, selected_node)

    def make_move(self, simulations, temp):
        self.run_mcts(simulations)
        move, pi = self.choose_move(temp)
        self.change_root(move)
        return move, pi
    
    #returns predicted value of game from perspective of current player, and expands the leaf node after
    #if the game is over at the given state, returns the value of the finished game from perspective of current player
    #this does a lot of stuff, maybe want to separate some parts out, at least for testing purposes
    def expand_and_evaluate(self, leaf: zero_mcts.ZeroNode):
        game_over = False

        value = 0
        if (leaf.board.outcome(claim_draw = True) is not None): #if game is over, we know the value
            game_over = True
            winner = leaf.board.outcome(claim_draw = True).winner
            if (winner is None):
                value = 0
            elif winner == leaf.board.turn:
                value = 1 #i don't think this is possible; e.g. if it is currently white's turn, the game cannot be won for white. keep because why not.
            else:
                value = -1
            return value
        
        board_array = np.array([btm.board_to_array(leaf.board)])
        prediction = self.model.predict(board_array, verbose=0)

        policy = prediction[0]
        policy = policy.reshape((64, 64))
        policy = np.exp(policy) #logits to probability part 1

        legal_moves = btm.legal_move_matrix(leaf.board)
        policy = policy*legal_moves #mask out illegal moves

        #normalize the policy matrix
        sum = np.sum(policy)
        policy = policy/sum #logits to probability part 2

        for move in leaf.board.legal_moves: #expanding the tree
            p = policy[move.from_square, move.to_square]
            copy_board = leaf.board.copy() #change stack to lower number of moves if this takes up too much memory (currently copies board w/ full move stack)
            copy_board.push(move)
            child = zero_mcts.ZeroNode(copy_board)

            edge = zero_mcts.ZeroEdge(leaf, child, move, p)
            leaf.add_edge(edge)
        
        value = prediction[1][0][0]
        #get eval from p
        eval = board_eval.eval_board(leaf.board)
        return value*constants.PREDICTION_WEIGHT+eval*(1-constants.PREDICTION_WEIGHT)
    
    def choose_move(self, temp):
        #get node visit proportions
        visit_nums = []
        total_visits = 0
        for edge in self.mcts.root.edges:
            visit_nums.append(edge.n)
            total_visits += edge.n
        
        visit_proportions = []
        for num in visit_nums:
            visit_proportions.append(num / total_visits)
        
        #now we build pi (flatten at end to be same shape as model output)
        pi = np.zeros((64, 64))
        for i, prop in enumerate(visit_proportions):
            move = self.mcts.root.edges[i].move
            pi[move.from_square, move.to_square] = prop
        pi = pi.flatten()

        #only infinitesimal temp and temp=1 get used, which is equivalent to choosing most visited and choosing the edge proportional to visits; so we implement this directly.
        if (temp == 0):
            maximizing_index = visit_proportions.index(max(visit_proportions))
            return self.mcts.root.edges[maximizing_index].move, pi
        else:
            rand = random.random()
            #print(rand)
            sum = 0
            #print(visit_proportions)
            for i, prob in enumerate(visit_proportions):
                sum += prob
                if sum >= rand:
                    return self.mcts.root.edges[i].move, pi
            #make sure we return some move, even though we should have already returned one
            #print(sum)
            return self.mcts.root.edges[-1].move, pi
    
    def change_root(self, move: chess.Move):
        #given some possible move from the current root position, change the root of the mcts to the board state after that move is played
        #if current root has any children, it will necessarily have the child that was passed into the method (all children are generated at the same time)
        if (not self.mcts.root.is_leaf()):
            for edge in self.mcts.root.edges:
                if edge.move == move:
                    self.mcts.root = edge.child
                    #print('updated root to existing node')
                    return
        cur_board = self.mcts.root.board
        cur_board.push(move)
        root_node = zero_mcts.ZeroNode(cur_board)
        self.mcts = zero_mcts.MCTS(root_node)
        #print('updated with new root')

        return