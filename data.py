from cmath import inf
from math import sqrt
import math
import random

import evaluator
import constants
import chess
import chess.pgn
import btm
import zeromodel
import zero_mcts
from player import Player

import numpy as np
import tensorflow as tf
from keras.models import Model

class Data:
    def __init__(self):
        self.game_data = [] #data from current game being played, gets moved to history at end of game
        self.history = [] #data from previous games, gets pickled every n games to save memory
    
    def add_move(self, turn, board_array, pi):
        dict = {'turn': turn, 'board': board_array, 'pi': pi}
        self.game_data.append(dict)
    
    def save_game(self, outcome: chess.Outcome): #once we know the outcome of the game, we can update the value for all data points and then save it
        self.update_value(outcome)
        self.history.extend(self.game_data)
        self.game_data = []
        #now slice history if it is longer than move_storage
        if len(self.history) > constants.MOVE_STORAGE:
            self.history = self.history[len(self.history) - constants.MOVE_STORAGE:]
    
    def update_value(self, outcome: chess.Outcome):
        winner = outcome.winner
        if winner is None:
            for move in self.game_data:
                move['z'] = 0
        else:
            for move in self.game_data:
                if move['turn'] == winner:
                    move['z'] = 1
                else:
                    move['z'] = -1
