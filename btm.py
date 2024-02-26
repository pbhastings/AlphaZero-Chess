from asyncio import constants
from copy import copy
import chess
import numpy as np
import constants

#returns number associated with each white piece by the chess library,
#negative the number for black pieces, 0 if piece is None
def get_piece_num(piece: chess.Piece):
    if (piece is None):
        return 0
    ret = piece.piece_type
    if (not piece.color):
        ret *= -1
    return ret

#current player pieces return values 0-5, other player pieces return values 6-11
#order is p,n,b,r,q,k, returns -1 if no piece
def get_piece_num_zero(piece: chess.Piece, turn: chess.Color):
    if (piece is None):
        return -1
    ret = piece.piece_type
    if (piece.color == turn):
        ret -= 1 #shift current player pieces from 1-6 to 0-5
    else:
        ret += 5 #shift current player pieces from 1-6 to 6-11
    return ret

#returns an 8x8 matrix where each element represents the piece on the corresponding
#square of the board; uses [-6,-1] for black, [1,6] for white, 0 for no piece
def board_to_matrix(board: chess.Board):
    board_matrix = np.zeros((8, 8), dtype=int)
    for i in range(64):
        p = board.piece_at(i)
        board_matrix[7 - (i // 8), i % 8] = get_piece_num(p)
    return board_matrix

#given a move, returns a sparse 64x64 matrix, 0's except for a 1 at
#row=square where piece moved from, column=square where piece moved to
def move_to_matrix(move: chess.Move):
    move_matrix = np.zeros((64, 64), dtype=int)
    move_matrix[move.from_square, move.to_square] = 1
    return move_matrix

#inverse of move_to_matrix
def matrix_to_move(arr):
    move_int = np.argmax(arr)
    from_square = move_int // 64
    to_square = move_int % 64
    move = chess.Move(from_square, to_square)
    return move

#sparse 64x64 matrix, for all legal moves, 1 at M_{from_square, to_square}
def legal_move_matrix(board: chess.Board):
    move_matrix = np.zeros((64, 64), dtype=int)
    for move in board.legal_moves:
        move_matrix[move.from_square, move.to_square] = 1
    return move_matrix

#converts board and some previous board states to sparse Nx8x8 array, used to train/predict
def board_to_array(board: chess.Board):
    #array for alphazero-like method:
    #   first 12xBOARD_MEMORY 8x8's represent position of different piece types; e.g. the first one is location of current player (P1) pawns
    #   boards are oriented in perspective of current player
    #   next 8x8 is ones if white's turn, zeros if black
    #   next two are P1 castling rights, next two are P2 castling rights
    copy_board = board.copy()
    turn = copy_board.turn
    num_layers = 12*constants.BOARD_MEMORY + 5
    arr = np.zeros((num_layers, 8, 8), dtype=int)

    #do color and castling rights planes before we pop moves off of board
    castle_index = 12*constants.BOARD_MEMORY
    if (turn):
        arr[castle_index] = np.ones((8, 8), dtype=int)
    if (copy_board.has_kingside_castling_rights(turn)):
        arr[castle_index + 1] = np.ones((8, 8), dtype=int)
    if (copy_board.has_queenside_castling_rights(turn)):
        arr[castle_index + 2] = np.ones((8, 8), dtype=int)
    if (copy_board.has_kingside_castling_rights(not turn)):
        arr[castle_index + 3] = np.ones((8, 8), dtype=int)
    if (copy_board.has_queenside_castling_rights(not turn)):
        arr[castle_index + 4] = np.ones((8, 8), dtype=int)


    for j in range(constants.BOARD_MEMORY):
        for i in range(64):
            p = copy_board.piece_at(i)
            if (p is not None):
                num = get_piece_num_zero(p, turn)
                #insert with correct orientation
                if (turn):
                    arr[12*j + num, 7 - (i // 8), i % 8] = 1
                else:
                    arr[12*j + num, i // 8, 7 - (i % 8)] = 1

        if (len(copy_board.move_stack) > 0):
            copy_board.pop()
        else:
            break
    
    return arr

#given a game outcome, 0 if draw, 1 if white wins, -1 if black wins
def outcome_to_value(outcome: chess.Outcome):
    if outcome.winner is None:
        return 0
    if outcome.winner:
        return 1
    return -1