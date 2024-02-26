import chess
import constants as c

import random

# converts square int which goes a1->b1->...->h8 to table index which goes a8->b8->...->h1
def square_to_index(square, color):
    file = square % 8
    rank = square // 8
    if (not color):
        file = 7 - file
        rank = 7 - rank
    index = file + (7 - rank) * 8
    return index

def eval_pawn(square, piece):
    return c.PAWN_VALUE + c.pawn_table[square_to_index(square, piece.color)]

def eval_knight(square, piece):
    return c.KNIGHT_VALUE + c.knight_table[square_to_index(square, piece.color)]

def eval_bishop(square, piece):
    return c.BISHOP_VALUE + c.bishop_table[square_to_index(square, piece.color)]

def eval_rook(square, piece):
    return c.ROOK_VALUE + c.rook_table[square_to_index(square, piece.color)]

def eval_queen(square, piece):
    return c.QUEEN_VALUE + c.queen_table[square_to_index(square, piece.color)]

def eval_king(square, piece):
    return c.KING_VALUE + c.king_table[square_to_index(square, piece.color)]

def eval_board(board):
    # if draw or checkmate, return corresponding evaluation
    outcome = board.outcome()
    if (outcome is not None):
        return eval_outcome(outcome)
    
    ev = 0
    for i in range(64):
        if (board.piece_at(i) is not None):
            ev += eval_piece(i, board.piece_at(i))
    return ev

def eval_outcome(outcome):
    # the checkmate values here should be less in abs value than the values in minimax so that even if a loss is forced, the engine picks a move
    if (outcome.winner is None):
        return 0
    else:
        if (outcome.winner):
            return 999998
        return -999998

def piece_value(piece: chess.Piece):
    piece_type = piece.piece_type

    if (piece_type == chess.PAWN):
        return c.PAWN_VALUE
    elif (piece_type == chess.KNIGHT):
        return c.KNIGHT_VALUE
    elif (piece_type == chess.BISHOP):
        return c.BISHOP_VALUE
    elif (piece_type == chess.ROOK):
        return c.ROOK_VALUE
    else:
        return c.QUEEN_VALUE

def eval_piece(square, piece):
    piece_type = piece.piece_type

    side = 1
    if (not piece.color):
        side = -1
    
    if (piece_type == chess.PAWN):
        return eval_pawn(square, piece) * side
    elif (piece_type == chess.KNIGHT):
        return eval_knight(square, piece) * side
    elif (piece_type == chess.BISHOP):
        return eval_bishop(square, piece) * side
    elif (piece_type == chess.ROOK):
        return eval_rook(square, piece) * side
    elif (piece_type == chess.QUEEN):
        return eval_queen(square, piece) * side
    else:
        return eval_king(square, piece) * side

def is_good_capture(move: chess.Move, board: chess.Board):
    if board.is_capture(move):
        capturer = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if (piece_value(captured) > piece_value(capturer)):
            return True
    return False


def sort_moves(moves, board: chess.Board):
    ordered_moves = []
    l = len(moves)
    for i in range(l):
        move = moves[l-i-1]
        if (board.is_capture(move) and not board.is_en_passant(move)):
            if (is_good_capture(move, board)):
                ordered_moves.append(move)
                moves.pop(l-i-1)

    l = len(moves)
    for i in range(l):
        move = moves[l-i-1]
        if (board.is_capture(move) and not board.is_en_passant(move)):
            ordered_moves.append(move)
            moves.pop(l-i-1)
    
    ordered_moves.extend(moves)

    return ordered_moves

def mc_eval(board: chess.Board):
    copy_board = board.copy(stack=False)
    while (not copy_board.is_game_over()):
        moves = list(copy_board.legal_moves)
        move = random.choice(moves)
        copy_board.push(move)
    
    if (copy_board.outcome().winner is None):
        return 0
    elif (copy_board.outcome().winner):
        return 1
    else:
        return -1