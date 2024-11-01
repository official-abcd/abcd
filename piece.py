'''
If a function has a # mark above the def statement, it is not being used.
'''

from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event
import re
import time
import sys

piece_values = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000
}

pst = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10,   0,   0,  0,  0,   0,   0, -10],
    [-10,   0,   5,  5,  5,   5,   0, -10],
    [-5,    0,   5,  5,  5,   5,   0,  -5],
    [0,     0,   5,  5,  5,   5,   0,  -5],
    [-10,   5,   5,  5,  5,   5,   0, -10],
    [-10,   0,   5,  0,  0,   0,   0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]
def is_path_clear(board, start_x, start_y, end_x, end_y):
    step_x = 1 if start_x < end_x else -1 if start_x > end_x else 0
    step_y = 1 if start_y < end_y else -1 if start_y > end_y else 0
    x, y = start_x + step_x, start_y + step_y
    while (x, y) != (end_x, end_y):
        if board[x][y] != " ":
            return False
        x += step_x
        y += step_y
    return True

def is_valid_move(board, start, end, player_color, last_move=None):
    start_x, start_y = start
    end_x, end_y = end
    if not (0 <= start_x < 8 and 0 <= start_y < 8 and 0 <= end_x < 8 and 0 <= end_y < 8):
        return False
    piece = board[start_x][start_y]
    if piece == " ":
        return False
    if (player_color == 'White' and piece.islower()) or (player_color == 'Black' and piece.isupper()):
        return False
    if board[end_x][end_y] != " " and is_same_color(piece, board[end_x][end_y]):
        return False

    def knight():
        return (abs(start_x - end_x) == 2 and abs(start_y - end_y) == 1) or \
               (abs(start_x - end_x) == 1 and abs(start_y - end_y) == 2)

    def rook():
        return (start_x == end_x or start_y == end_y) and is_path_clear(board, start_x, start_y, end_x, end_y)

    def bishop():
        return abs(start_x - end_x) == abs(start_y - end_y) and is_path_clear(board, start_x, start_y, end_x, end_y)

    def queen():
        return (rook() or bishop())

    def king():
        if abs(start_y - end_y) == 2 and start_x == end_x:
            if start_y < end_y:
                if is_path_clear(board, start_x, start_y, start_x, 7) and board[start_x][7].lower() == 'r':
                    return True
            else:
                if is_path_clear(board, start_x, start_y, start_x, 0) and board[start_x][0].lower() == 'r':
                    return True
        return max(abs(start_x - end_x), abs(start_y - end_y)) == 1

    def pawn():
        direction = -1 if piece.isupper() else 1
        start_row = 6 if piece.isupper() else 1
        if start_y == end_y:
            if start_x + direction == end_x and board[end_x][end_y] == " ":
                return True
            if start_x == start_row and start_x + 2 * direction == end_x and board[start_x + direction][end_y] == " " and board[end_x][end_y] == " ":
                return True
        if abs(start_y - end_y) == 1 and end_x == start_x + direction:
            if board[end_x][end_y] != " " and (board[end_x][end_y].islower() if piece.isupper() else board[end_x][end_y].isupper()):
                return True
            if last_move:
                last_start, last_end = last_move
                last_start_row, last_start_col = last_start
                last_end_row, last_end_col = last_end
                if board[last_end_row][last_end_col].lower() == 'p' and abs(last_start_row - last_end_row) == 2 and last_end_col == end_y:
                    if end_x == last_end_row + direction:
                        return True
        return False

    valid_move = False
    if piece.lower() == 'k':
        valid_move = king()
    elif piece.lower() == 'q':
        valid_move = queen()
    elif piece.lower() == 'r':
        valid_move = rook()
    elif piece.lower() == 'b':
        valid_move = bishop()
    elif piece.lower() == 'n':
        valid_move = knight()
    elif piece.lower() == 'p':
        valid_move = pawn()
    if valid_move:                                                         
        temp_board = [row[:] for row in board]                          
        temp_board[end_x][end_y] = piece
        temp_board[start_x][start_y] = " "                                  
        check_status = is_check(temp_board)
        if (piece.isupper() and check_status["white"]) or (piece.islower() and check_status["black"]):
            return False
    return valid_move                                                       

def is_check(board):
    white_king_pos = find_king(board, 'white')
    black_king_pos = find_king(board, 'black')
    check_status = {"white": False, "black": False}
    if not white_king_pos or not black_king_pos:
        return check_status
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece != " " and piece.islower():
                if is_valid_move(board, (x, y), white_king_pos, 'black'):
                    check_status["white"] = True
                    break
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece != " " and piece.isupper():
                if is_valid_move(board, (x, y), black_king_pos, 'white'):
                    check_status["black"] = True
                    break
    return check_status

def is_checkmate(board, color):
    if not is_check(board)[color]:
        return False
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece != " " and ((color == 'white' and piece.isupper()) or (color == 'black' and piece.islower())):
                for dx in range(8):
                    for dy in range(8):
                        if is_valid_move(board, (x, y), (dx, dy), color):
                            temp_board = [row[:] for row in board]
                            temp_board[dx][dy] = piece
                            temp_board[x][y] = " "
                            if not is_check(temp_board)[color]:
                                return False
    return True

def is_stalemate(board, color):
    if is_check(board)[color]:
        return False
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece == " ":
                continue
            if (color == 'white' and piece.isupper()) or (color == 'black' and piece.islower()):
                for dx in range(8):
                    for dy in range(8):
                        if is_valid_move(board, (x, y), (dx, dy), color):
                            temp_board = [row[:] for row in board]
                            temp_board[dx][dy] = piece
                            temp_board[x][y] = " "
                            if not is_check(temp_board)[color]:
                                return False
    return True

def find_king(board, color):
    king = 'K' if color == 'white' else 'k'
    for x in range(8):
        for y in range(8):
            if board[x][y] == king:
                return (x, y)
    return None

def is_same_color(piece1, piece2):
    return (piece1.isupper() and piece2.isupper()) or (piece1.islower() and piece2.islower())

def get_all_moves(board, player_color="Black", last_move=None):
    moves = []
    for start_x in range(8):
        for start_y in range(8):
            piece = board[start_x][start_y]
            if piece == " ":
                continue
            if (player_color == 'White' and piece.islower()) or (player_color == 'Black' and piece.isupper()):
                continue
            piece_moves = generate_piece_moves(board, piece, start_x, start_y, last_move)
            for end_x, end_y in piece_moves:
                if is_valid_move(board, (start_x, start_y), (end_x, end_y), player_color, last_move):
                    moves.append(((start_x, start_y), (end_x, end_y)))
    return moves

###
def get_tree(board, player_color, depth, max_depth, last_move=None):
    if depth == max_depth:
        return []
    all_moves = get_all_moves(board, player_color, last_move)
    tree = []
    next_color = 'White' if player_color == 'Black' else 'Black'
    for move in all_moves:
        start, end = move
        temp_board = [row[:] for row in board]
        temp_board[end[0]][end[1]] = temp_board[start[0]][start[1]]
        temp_board[start[0]][start[1]] = " "
        subtree = get_tree(temp_board, next_color, depth + 1, max_depth, move)
        tree.append((move, subtree))
    return tree

###
def undo_move(board, start, end, captured_piece=None, special_move=None):
    start_x, start_y = start
    end_x, end_y = end
    board[start_x][start_y] = board[end_x][end_y]
    board[end_x][end_y] = " "
    if captured_piece:
        board[end_x][end_y] = captured_piece
    if special_move == 'castling':
        if end_y - start_y == 2:
            board[start_x][7] = board[start_x][end_y - 1]
            board[start_x][end_y - 1] = " "
        elif start_y - end_y == 2:
            board[start_x][0] = board[start_x][end_y + 1]
            board[start_x][end_y + 1] = " "
    elif special_move == 'en_passant':
        if board[start_x][start_y].lower() == 'p':
            direction = 1 if board[start_x][start_y].isupper() else -1
            board[end_x - direction][end_y] = "p" if direction == 1 else "P"
    elif special_move == 'promotion':
        if board[start_x][start_y].lower() in ['q', 'r', 'b', 'n']:
            board[start_x][start_y] = 'P' if board[start_x][start_y].isupper() else 'p'
    return board

def generate_piece_moves(board, piece, x, y, last_move):
    moves = []
    directions = []
    
    # Pawn moves
    if piece.lower() == 'p':
        direction = -1 if piece.isupper() else 1
        start_row = 6 if piece.isupper() else 1
        if 0 <= x + direction < 8 and board[x + direction][y] == " ":
            moves.append((x + direction, y))
            if x == start_row and board[x + 2 * direction][y] == " ":
                moves.append((x + 2 * direction, y))
        for dy in [-1, 1]:
            nx, ny = x + direction, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if board[nx][ny] != " ":
                    moves.append((nx, ny))
                elif last_move:
                    last_start, last_end = last_move
                    if (board[last_end[0]][last_end[1]].lower() == 'p' and 
                        abs(last_start[0] - last_end[0]) == 2 and 
                        last_end[0] == x and last_end[1] == ny):
                        moves.append((nx, ny))

    # Knight moves
    elif piece.lower() == 'n':
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dx, dy in knight_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                moves.append((nx, ny))

    # Bishop moves
    elif piece.lower() == 'b':
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Rook moves
    elif piece.lower() == 'r':
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Queen moves (combination of bishop and rook)
    elif piece.lower() == 'q':
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]

    # King moves
    elif piece.lower() == 'k':
        king_moves = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in king_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                moves.append((nx, ny))
        
        # Castling moves (requires boundary checking)
        if not is_check(board)["white"] or is_check(board)["black"]:
            # Kingside castling
            if y + 3 < 8:
                if (board[x][y+1] == ' ' and 
                    board[x][y+2] == ' ' and 
                    board[x][y+3].lower() == 'r'):
                    moves.append((x, y+2))
            # Queenside castling
            if y - 4 >= 0:
                if (board[x][y-1] == ' ' and 
                    board[x][y-2] == ' ' and 
                    board[x][y-3] == ' ' and 
                    board[x][y-4].lower() == 'r'):
                    moves.append((x, y-2))

    # Sliding moves for bishop, rook, and queen
    if directions:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                moves.append((nx, ny))
                if board[nx][ny] != " ":
                    break
                nx += dx
                ny += dy
                
    return moves

def generate_attacked_squares(board, piece, x, y):
    '''
    Generates squares attacked by the piece at position (x, y).
    '''
    attacked_squares = []
    directions = []

    # Pawn attacks
    if piece.lower() == 'p':
        direction = -1 if piece.isupper() else 1
        for dy in [-1, 1]:
            nx, ny = x + direction, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                attacked_squares.append((nx, ny))

    # Knight attacks
    elif piece.lower() == 'n':
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        for dx, dy in knight_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                attacked_squares.append((nx, ny))

    # Bishop directions
    elif piece.lower() == 'b':
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Rook directions
    elif piece.lower() == 'r':
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Queen directions
    elif piece.lower() == 'q':
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                      (-1, 0), (1, 0), (0, -1), (0, 1)]

    # King attacks
    elif piece.lower() == 'k':
        king_moves = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                      (-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in king_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                attacked_squares.append((nx, ny))

    # Sliding pieces (bishop, rook, queen)
    if directions:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if board[nx][ny] != " ":
                    if not is_same_color(piece, board[nx][ny]):
                        attacked_squares.append((nx, ny))
                    break
                attacked_squares.append((nx, ny))
                nx += dx
                ny += dy

    return attacked_squares

def get_attacked_squares(board, player_color):
    """
    Generates a set of all squares attacked by the given player_color.
    """
    attacked_squares = set()
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece != " ":
                if (player_color == 'White' and piece.isupper()) or (player_color == 'Black' and piece.islower()):
                    attacked = generate_attacked_squares(board, piece, x, y)
                    attacked_squares.update(attacked)
    return attacked_squares


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def is_game_over(board):
    return is_checkmate(board, 'white') or is_checkmate(board, 'black') or \
           is_stalemate(board, 'white') or is_stalemate(board, 'black')

def convert_move_to_uci(start_pos, end_pos, board, promotion=None, last_move=None):
    """
    Converts an internal move representation to UCI format,
    handling promotions, castling, and en passant.
    """
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    piece = board[start_row][start_col]
    moved_piece = piece.lower()
    promotion_piece = ''
    is_castling = False
    is_en_passant = False

    # Determine if the move is a pawn promotion
    if moved_piece == 'p':
        if (piece.isupper() and end_row == 0) or (piece.islower() and end_row == 7):
            # If promotion piece is provided, use it; default to queen
            if promotion:
                promotion_piece = promotion.lower()
            else:
                promotion_piece = 'q'  # Default to queen promotion

    # Determine if the move is castling
    if moved_piece == 'k':
        if piece.isupper() and start_pos == (7, 4):  # White king's initial position
            if end_pos == (7, 6):  # Kingside castling
                is_castling = True
            elif end_pos == (7, 2):  # Queenside castling
                is_castling = True
        elif piece.islower() and start_pos == (0, 4):  # Black king's initial position
            if end_pos == (0, 6):  # Kingside castling
                is_castling = True
            elif end_pos == (0, 2):  # Queenside castling
                is_castling = True

    # Determine if the move is an en passant capture
    if moved_piece == 'p' and board[end_row][end_col] == ' ':
        if abs(start_col - end_col) == 1 and abs(start_row - end_row) == 1:
            if last_move:
                last_start, last_end = last_move
                last_piece = board[last_end[0]][last_end[1]]
                if last_piece.lower() == 'p' and abs(last_start[0] - last_end[0]) == 2:
                    if last_end[0] == start_row and last_end[1] == end_col:
                        is_en_passant = True

    # Convert positions to UCI notation
    start_file = chr(ord('a') + start_col)
    start_rank = str(8 - start_row)
    end_file = chr(ord('a') + end_col)
    end_rank = str(8 - end_row)
    uci_move = f"{start_file}{start_rank}{end_file}{end_rank}{promotion_piece}"

    return uci_move
        
def parse_fen(fen_str):
    board = []
    fen_parts = fen_str.strip().split()
    rows = fen_parts[0].split('/')
    for fen_row in rows:
        board_row = []
        for c in fen_row:
            if c.isdigit():
                board_row.extend([' '] * int(c))
            else:
                board_row.append(c)
        board.append(board_row)
    while len(board) < 8:
        board.append([' '] * 8)
    return board

def parse_uci_move(move_str):
    if len(move_str) < 4:
        return None
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
    try:
        start_col = col_map[move_str[0]]
        start_row = 8 - int(move_str[1])
        end_col = col_map[move_str[2]]
        end_row = 8 - int(move_str[3])
        return ((start_row, start_col), (end_row, end_col))
    except (KeyError, ValueError):
        return None

def go_loop(searcher, hist, stop_event, max_movetime=0, max_depth=0, debug=False):
    import main
    if debug:
        print(f"Going movetime={max_movetime}, depth={max_depth}")
    start = time.time()
    best_move = None
    for i, j in main.search(hist, 18, 1.0):
            depth = j
            score = main.evaulate(hist)
            print("info depth", depth, "score cp", score, "pv", move_str)
    my_pv = pv(searcher, hist[-1], include_scores=False)
    print("bestmove", my_pv[0] if my_pv else "(none)")

def run(startpos):
    #import main
    debug = False
    hist = [startpos]
    with ThreadPoolExecutor(max_workers=1) as executor:
        go_future = executor.submit(lambda: None)
        do_stop_event = Event()
        while True:
            try:
                args = input().split()
                if not args:
                    continue
                elif args[0] in ("stop", "quit"):
                    if go_future.running():
                        if debug:
                            print("Stopping go loop...")
                        do_stop_event.set()
                        go_future.result()
                    else:
                        if debug:
                            print("Go loop not running...")
                    if args[0] == "quit":
                        break
                elif not go_future.done():
                    print(f"Ignoring input {args}. Please call 'stop' first.")
                    continue
                go_future.result(timeout=0)
                if args[0] == "uci":
                    print(f"id name {version}")
                    print("uciok")
                elif args[0] == "setoption":
                    _, uci_key, _, uci_value = args[1:]
                    setattr(sys.modules[__name__], uci_key, int(uci_value))
                elif args[0] == "isready":
                    print("readyok")
                elif args[0] == "quit":
                    break
                elif args[:2] == ["position", "startpos"]:
                    hist = [startpos]
                    for ply, move in enumerate(args[3:]):
                        hist.append(hist[-1].move(parse_move(move, ply % 2 == 0)))
                elif args[:2] == ["position", "fen"]:
                    pos = parse_fen(*args[2:8])
                    hist = [pos] if get_color(pos) == WHITE else [rotate(pos), pos]
                    if len(args) > 8:
                        assert args[8] == "moves"
                        for move in args[9:]:
                            hist.append(hist[-1].move(parse_move(move, len(hist) % 2 == 1)))
                elif args[0] == "go":
                    think = 10**6
                    max_depth = 100
                    loop = go_loop
                    if args[1:] == [] or args[1] == "infinite":
                        pass
                    elif args[1] == "movetime":
                        movetime = args[2]
                        think = int(movetime) / 1000
                    elif args[1] == "wtime":
                        wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
                        if len(hist) % 2 == 0:
                            wtime, winc = btime, binc
                        think = min(wtime / 40 + winc, wtime / 2 - 1)
                        if len(hist) < 3:
                            think = min(think, 1)
                    elif args[1] == "depth":
                        max_depth = int(args[2])
                    elif args[1] in ("mate", "draw"):
                        max_depth = int(args[2])
                        loop = partial(mate_loop, find_draw=args[1] == "draw")
                    elif args[1] == "perft":
                        perft(hist[-1], int(args[2]), debug=debug)
                        continue
                    do_stop_event.clear()
                    go_future = executor.submit(
                        loop,
                        #searcher,
                        hist,
                        do_stop_event,
                        think,
                        max_depth,
                        debug=debug,
                    )
                    def callback(fut):
                        fut.result(timeout=0)
                    go_future.add_done_callback(callback)
            except (KeyboardInterrupt, EOFError):
                if go_future.running():
                    if debug:
                        print("Stopping go loop...")
                    do_stop_event.set()
                    go_future.result()
                break

def evaluate_pawn_structure(board):
    score = 0
    for color in ['white', 'black']:
        pawn_positions = []
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if (color == 'white' and piece == 'P') or (color == 'black' and piece == 'p'):
                    pawn_positions.append((x, y))
        # Isolated Pawns
        for pawn in pawn_positions:
            x, y = pawn
            files_to_check = []
            if y > 0:
                files_to_check.append(y - 1)
            if y < 7:
                files_to_check.append(y + 1)
            isolated = True
            for fy in files_to_check:
                for fx in range(8):
                    if (fx, fy) in pawn_positions:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                penalty = 15
                score -= penalty if color == 'white' else -penalty
        # Doubled Pawns
        file_counts = {}
        for pawn in pawn_positions:
            x, y = pawn
            file_counts.setdefault(y, []).append(x)
        for file_pawns in file_counts.values():
            if len(file_pawns) > 1:
                penalty = 10 * (len(file_pawns) - 1)
                score -= penalty if color == 'white' else -penalty
        # Passed Pawns
        for pawn in pawn_positions:
            x, y = pawn
            passed = True
            if color == 'white':
                for fx in range(x):
                    if board[fx][y] == 'p':
                        passed = False
                        break
            else:
                for fx in range(x + 1, 8):
                    if board[fx][y] == 'P':
                        passed = False
                        break
            if passed:
                advancement = (7 - x) * 10 if color == 'white' else x * 10
                score += advancement if color == 'white' else -advancement
    return score

def evaluate_king_safety(board):
    score = 0
    for color in ['white', 'black']:
        king_pos = find_king(board, color)
        if not king_pos:
            continue
        x, y = king_pos
        pawn_shield = 0
        if color == 'white':
            for dx in [-1, 0, 1]:
                nx, ny = x - 1, y + dx
                if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == 'P':
                    pawn_shield += 1
        else:
            for dx in [-1, 0, 1]:
                nx, ny = x + 1, y + dx
                if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == 'p':
                    pawn_shield += 1
        if pawn_shield < 2:
            penalty = (2 - pawn_shield) * 20
            score -= penalty if color == 'white' else -penalty
    return score

def evaluate_bishop_pair(board):
    score = 0
    for color in ['white', 'black']:
        bishop_count = sum(
            1 for x in range(8) for y in range(8)
            if (board[x][y] == 'B' and color == 'white') or (board[x][y] == 'b' and color == 'black')
        )
        if bishop_count >= 2:
            bonus = 50
            score += bonus if color == 'white' else -bonus
    return score

def evaluate(board, phase=None):
    score = 0
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece == " ":
                continue
            piece_value = piece_values.get(piece, 0)
            ps_value = pst[x][y]
            if piece == 'k'.lower():
                if x != 7 or x != 0:
                    if y not in [1, 2, 5, 6] and phase != None:
                        ps_value -= 100000000000000000000000000
            if piece.isupper():
                score += piece_value + ps_value
            else:
                score -= piece_value + ps_value
    white_moves = len(get_all_moves(board, 'White'))
    black_moves = len(get_all_moves(board, 'Black'))
    mobility_weight = 10
    score += mobility_weight * (white_moves - black_moves)
    score += evaluate_pawn_structure(board)
    score += evaluate_king_safety(board)
    score += evaluate_bishop_pair(board)
    if is_checkmate(board, 'white'):
        return -999999
    if is_checkmate(board, 'black'):
        return 999999
    if is_stalemate(board, 'white') or is_stalemate(board, 'black'):
        return 0
    return score
