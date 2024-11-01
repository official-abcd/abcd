import random
import pickle
import threading
import time
import math
from queue import Queue
from piece import *
openings = {
    "e2 e4": "c7 c5",  # Sicilian Defense
    "d2 d4": "g8 f6",  # Indian Game
    "g1 f3": "d7 d5",  # Reti Opening
    "c2 c4": "g8 f6"   # English Opening
}
BOARD_SIZE = 8
depth = 23
pondering_move = None
pondering_thread = None
predict_opponent_move = None
max_time = 0.8
q_table_lock = threading.Lock()
pondering_stop_event = threading.Event()
center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
unicode_pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    ' ': ' '
}
piece_values = {
        'p': 1, 'n': 3, 'b': 3.1, 'r': 5, 'q': 9, 'k': 0,
        'P': 1, 'N': 3, 'B': 3.1, 'R': 5, 'Q': 9, 'K': 0
    }
initial_board = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]
try:
    with open("data.dat", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.2
def print_board(board, is_white=True):
    '''
    prints the board

    pros:
    works properly
    cons:
    is a list of lists:
    slow to loop through every board.
    consider:
    Use bitboards?
    '''
    if is_white:
        print("  a b c d e f g h")
        for row in range(BOARD_SIZE):
            print(8 - row, end=" ")
            for col in range(BOARD_SIZE):
                print(unicode_pieces[board[row][col]], end=" ")
            print(8 - row)
        print("  a b c d e f g h")
    else:
        board.reverse()
        print("  h g f e d c b a")
        for row in range(BOARD_SIZE):
            print(row + 1, end=" ")
            for col in range(BOARD_SIZE):
                print(unicode_pieces[board[row][col]], end=' ')
            print(row + 1)
        print("  h g f e d c b a")
        rotate(board)
    return board
def move_piece(board, start, end, last_move, sim_move=False):
    '''
    moves the piece

    pros:
    works properly, computes
    special moves.
    cons:
    too long, could return
    board.
    consider:
    Could return board instead of having
    it as a semi-global list of lists.
    '''
    start_row, start_col = start
    end_row, end_col = end
    piece = board[start_row][start_col]
    if piece == " ":
        if not sim_move:
            print("Invalid move: no piece at start position")
        return False, last_move
    if piece.lower() == 'p':
        if abs(start_row - end_row) == 1 and abs(start_col - end_col) == 1 and board[end_row][end_col] == ' ':
            if last_move:
                last_start, last_end = last_move
                if board[last_end[0]][last_end[1]].lower() == 'p' and abs(last_start[0] - last_end[0]) == 2 and last_end[0] == start_row and last_end[1] == end_col:
                    board[last_end[0]][last_end[1]] = ' '
    if piece.lower() == 'k' and abs(start_col - end_col) == 2 and start_row == end_row:
        check_status = is_check(board)
        if (piece.isupper() and check_status["white"]) or (piece.islower() and check_status["black"]):
            return False, last_move
        if end_col > start_col:
            if is_path_clear(board, start_row, start_col, start_row, 7) and board[start_row][7].lower() == 'r':
                for col in range(start_col + 1, end_col + 1):
                    temp_board = [row[:] for row in board]
                    temp_board[start_row][start_col] = " "
                    temp_board[start_row][col] = piece
                    check_status = is_check(temp_board)
                    if (piece.isupper() and check_status["white"]) or (piece.islower() and check_status["black"]):
                        return False, last_move
                board[start_row][end_col] = piece
                board[start_row][start_col] = " "
                board[start_row][end_col - 1] = board[start_row][7]
                board[start_row][7] = " "
                return True, (start, end)
        else:
            if is_path_clear(board, start_row, start_col, start_row, 0) and board[start_row][0].lower() == 'r':
                for col in range(start_col - 1, end_col - 1, -1):
                    temp_board = [row[:] for row in board]
                    temp_board[start_row][start_col] = " "
                    temp_board[start_row][col] = piece
                    check_status = is_check(temp_board)
                    if (piece.isupper() and check_status["white"]) or (piece.islower() and check_status["black"]):
                        return False, last_move
                board[start_row][end_col] = piece
                board[start_row][start_col] = " "
                board[start_row][end_col + 1] = board[start_row][0]
                board[start_row][0] = " "
                return True, (start, end)
    temp_board = [row[:] for row in board]
    temp_board[end_row][end_col] = piece
    temp_board[start_row][start_col] = " "
    current_check_status = is_check(temp_board)
    if (piece.isupper() and current_check_status["white"]) or (piece.islower() and current_check_status["black"]):
        if not sim_move:
            print("Invalid move: move puts player in check")
        return False, last_move
    board[end_row][end_col] = piece
    board[start_row][start_col] = " "
    new_check_status = is_check(board)
    if new_check_status["white"] or new_check_status["black"]:
        if not sim_move:
            print("Check!")
    return True, (start, end)
def parse_move(move):
    '''
    Changes algabraic notation into
    cartesian coordinates tuple of
    tuples.

    pros:
    Actually works
    cons:
    The board is a list of lists
    consider:
    Change the board into bit board?
    '''
    move = move.replace(',', '')
    parts = move.split()
    if len(parts) != 2 or not all(len(part) == 2 for part in parts):
        return None
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    try:
        start_col, start_row = col_map[parts[0][0]], int(parts[0][1]) - 1
        end_col, end_row = col_map[parts[1][0]], int(parts[1][1]) - 1
        return ((7 - start_row, start_col), (7 - end_row, end_col))
    except (KeyError, ValueError):
        return None
def handle_promotion(board, x, y, piece):
    '''
    Checks if promotion needed, if so then
    prompt the user for which piece the pawn
    will promote to.

    pros:
    works
    cons:
    consider:
    '''
    if (piece == 'P' and x == 0) or (piece == 'p' and x == 7):
        while True:
            promotion = input("Promote to (q/r/b/n): ").lower()
            if promotion in ['q', 'r', 'b', 'n']:
                break
            print("Invalid choice. Choose q, r, b, or n.")
        board[x][y] = promotion.upper() if piece.isupper() else promotion
def get_q_value(state, action):
    state_tuple = tuple(tuple(row) for row in state)
    return q_table.get((state_tuple, action), 0)
def update_q_table(state, action, reward, next_state):
    '''
    Self explanitory

    pros:
    works
    cons:'''
    state_tuple = tuple(tuple(row) for row in state)
    next_state_tuple = tuple(tuple(row) for row in next_state)
    current_q = get_q_value(state, action)
    next_q_values = [get_q_value(next_state, a) for a in get_possible_actions(next_state)]
    max_next_q = max(next_q_values, default=0)
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[(state_tuple, action)] = new_q
def get_possible_actions(state):
    '''
    Gets all possible actions.
    
    pros:
    Works, less params
    cons:
    There are two functions that
    do the exact same thing! this
    function and get_all_moves.
    consider:
    Choosing one function.
    '''
    actions = []
    for start_row in range(BOARD_SIZE):
        for start_col in range(BOARD_SIZE):
            if state[start_row][start_col].islower():
                for end_row in range(BOARD_SIZE):
                    for end_col in range(BOARD_SIZE):
                        if is_valid_move(state, (start_row, start_col), (end_row, end_col), "Black", None):
                            actions.append(((start_row, start_col), (end_row, end_col)))
    return actions
# Not used functions: #
###
def prune(tree, board, player_color="Black", depth=0):
    good_positions = []
    for position in tree:
        move, subtree = position
        start, end = move
        if not (0 <= start[0] < 8 and 0 <= start[1] < 8 and 0 <= end[0] < 8 and 0 <= end[1] < 8):
            continue
        piece_moved = board[start[0]][start[1]]
        piece_captured = board[end[0]][end[1]]

        # Make the move
        board[end[0]][end[1]] = piece_moved
        board[start[0]][start[1]] = " "

        if not subtree:
            value = evaluate(board)
            if (player_color == 'White' and value > 1.0) or (player_color == 'Black' and value < -1.0):
                good_positions.append((move, value))
        else:
            pruned_subtree = prune(subtree, board, 'White' if player_color == 'Black' else 'Black', depth + 1)
            if pruned_subtree:
                good_positions.append((move, pruned_subtree))

        # Undo the move
        board[start[0]][start[1]] = piece_moved
        board[end[0]][end[1]] = piece_captured

    return good_positions
###
def ponder(board, depth=depth):
    """
    Function to handle pondering, calculating potential future positions
    and directly updating the q_table during the opponent's time.
    """
    global predicted_opponent_moves
    pondering_stop_event.clear()
    opponent_moves = get_all_moves(board, 'White')
    opponent_moves = order_moves(board, opponent_moves, 'White')
    predicted_opponent_moves = opponent_moves[:5]  # Consider top 5 potential moves
    for predicted_move in predicted_opponent_moves:
        if pondering_stop_event.is_set():
            break
        temp_board = [row[:] for row in board]
        move_piece(temp_board, predicted_move[0], predicted_move[1], None, sim_move=True)
        simulate_future_positions(temp_board, depth, max_moves=5)  # Pass max_moves
###
def start_pondering(board):
    global pondering_thread
    if pondering_thread and pondering_thread.is_alive():
        stop_pondering()
    pondering_thread = threading.Thread(target=ponder, args=(board,))
    pondering_thread.start()
###
def stop_pondering():
    """
    Stop the pondering process if the player's move doesn't match the pondered line.
    """
    global pondering_thread
    pondering_stop_event.set()
    if pondering_thread and pondering_thread.is_alive():
        pondering_thread.join()
    pondering_thread = None
###
def simulate_future_positions(board, depth, player_color='Black', last_move=None, max_moves=5, visited_positions=None):
    """
    Simulate future positions to a certain depth and update the q_table directly.
    """
    if visited_positions is None:
        visited_positions = set()
    state_tuple = tuple(tuple(row) for row in board)
    if state_tuple in visited_positions:
        return
    visited_positions.add(state_tuple)
    if depth == 0 or is_game_over(board):
        with q_table_lock:
            if state_tuple not in q_table:
                q_table[state_tuple] = evaluate(board)
        return
    player_color = 'White' if player_color == 'Black' else 'Black'
    possible_moves = get_all_moves(board, player_color, last_move)
    ordered_moves = order_moves(board, possible_moves, player_color, last_move)
    limited_moves = ordered_moves[:max_moves]  # Limit to top N moves
    for move in limited_moves:
        temp_board = [row[:] for row in board]
        move_piece(temp_board, move[0], move[1], last_move, sim_move=True)
        simulate_future_positions(temp_board, depth - 1, player_color, move, max_moves, visited_positions)

def minimax(board, last_move, depth, max_depth, alpha, beta, maximizing_player):
    """
    Minimax function with alpha-beta pruning and iterative deepening.
    """
    best_move = None
    player_color = 'White' if maximizing_player else 'Black'
    possible_moves = get_all_moves(board, player_color, last_move)
    possible_moves = order_moves(board, possible_moves, player_color, last_move)
    good_quick_moves = [possible_moves[i] for i in range(int(len(possible_moves) / 2))]
    state_tuple = tuple(tuple(row) for row in board)
    if state_tuple in q_table and state_tuple in good_quick_moves:
        return q_table[state_tuple], None
    if depth == max_depth or is_game_over(board):
        score = evaluate(board)
        if not maximizing_player:
            score = -score  # Invert score for minimizing player
        q_table[state_tuple] = score
        return score, None
    if maximizing_player:
        value = float('-inf')
        for move_score, move in enumerate(possible_moves):
            if move_score < len(possible_moves) / 2:
                continue
            start_pos, end_pos = move
            piece_moved = board[start_pos[0]][start_pos[1]]
            captured_piece = board[end_pos[0]][end_pos[1]]
            board[end_pos[0]][end_pos[1]] = piece_moved
            board[start_pos[0]][start_pos[1]] = " "
            eval_score, _ = minimax(board, move, depth + 1, max_depth, alpha, beta, False)
            board[start_pos[0]][start_pos[1]] = piece_moved
            board[end_pos[0]][end_pos[1]] = captured_piece
            #if eval_score > value:
            #    value = eval_score
            #    best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        q_table[state_tuple] = value
        return value, best_move
    else:
        value = float('inf')
        for move_score, move in enumerate(possible_moves):
            if move_score > len(possible_moves) / 2:
                continue
            start_pos, end_pos = move
            piece_moved = board[start_pos[0]][start_pos[1]]
            captured_piece = board[end_pos[0]][end_pos[1]]
            board[end_pos[0]][end_pos[1]] = piece_moved
            board[start_pos[0]][start_pos[1]] = " "
            eval_score, _ = minimax(board, move, depth + 1, max_depth, alpha, beta, True)
            board[start_pos[0]][start_pos[1]] = piece_moved
            board[end_pos[0]][end_pos[1]] = captured_piece
            #if eval_score < value:
            #    value = eval_score
            #    best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        q_table[state_tuple] = value
        return value, best_move
def search(board, max_depth=depth, time_limit=max_time, computer_color="Black"):
    """
    Use iterative deepening to determine the best move, with a time limit.
    Note that to the engine, it almosts treats the time_limit like a sugge
    -stion, sometimes 7 seconds to 2 seconds!
    """
    start_time = time.time()
    best_move = None
    current_depth = 0
    alpha = float('-inf')
    beta = float('inf')
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            current_depth = depth # capture the depth!!
            break  # Exit if time limit exceeded
        if computer_color == "Black":
            score, move = minimax(board, None, 0, depth, alpha, beta, False)
        else:
            score, move = minimax(board, None, 0, depth, alpha, beta, True)
        if move:
            best_move = move
    return best_move, current_depth
def choose_action(state):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(get_possible_actions(state))
    q_values = {action: get_q_value(state, action) for action in get_possible_actions(state)}
    max_q = max(q_values.values(), default=0)
    actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
    return random.choice(actions_with_max_q)
def order_moves(board, moves, player_color, last_move=None, killer_moves=None, history_table=None):
    """
    Improved move ordering function using MVV/LVA heuristic, killer moves, and history heuristic.
    """
    if killer_moves is None:
        killer_moves = {}
    if history_table is None:
        history_table = {}
    move_scores = []
    opponent_color = 'White' if player_color == 'Black' else 'Black'
    opponent_attacked_squares = get_attacked_squares(board, opponent_color)
    our_defended_squares = get_attacked_squares(board, player_color)
    opponent_king_pos = find_king(board, opponent_color.lower())
    for move in moves:
        start_pos, end_pos = move
        piece = board[start_pos[0]][start_pos[1]]
        target_piece = board[end_pos[0]][end_pos[1]]
        piece_value = piece_values.get(piece.lower(), 0)
        target_value = piece_values.get(target_piece.lower(), 0)
        score = 0
        if target_piece != " " and not is_same_color(piece, target_piece):
            mvv_lva_score = (target_value * 10) - piece_value
            score += 100000 + mvv_lva_score  # High base score for captures
        killer_key = (depth, player_color)
        if killer_moves.get(killer_key) == move:
            score += 90000  # High score for killer moves
        history_score = history_table.get((piece, move), 0)
        score += history_score
        if end_pos in center_squares:
            score += 500  # Encourage central control
        if end_pos in opponent_attacked_squares:
            if end_pos not in our_defended_squares:
                score -= 5000
        if piece.lower() == 'p':
            if (piece.isupper() and end_pos[0] == 0) or (piece.islower() and end_pos[0] == 7):
                score += 800
        if opponent_king_pos and is_valid_move(board, start_pos, opponent_king_pos, player_color):
            score += 300
        q_value = get_q_value(board, move)
        score += q_value
        move_scores.append((score, move))
    move_scores.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [move for score, move in move_scores]
    return ordered_moves
def play_chess():
    '''
Main game to play the chess'''
    board = [row[:] for row in initial_board]
    player_color = 'White'
    last_move = None
    mode = input("Do you want to play against a person or the computer? ").strip().lower()
    if mode in ["person", "play against a person"]:
        while True:
            print_board(board)
            move = input(f"{player_color}'s turn. Enter your move (e.g., e2 e4): ").strip()
            if move.lower() == 'quit':
                with open("data.dat", "wb") as file:
                        pickle.dump(q_table, file)
                return
            parsed_move = parse_move(move)
            if parsed_move is None:
                print("Invalid input. Please enter your move in the format 'e2 e4'.")
                continue
            start_pos, end_pos = parsed_move
            if is_valid_move(board, start_pos, end_pos, player_color, last_move):
                valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                if valid_move:
                    handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                    if is_checkmate(board, 'black' if player_color == 'White' else 'white'):
                        print_board(board)
                        print(f"Checkmate! {player_color} wins!")
                        with open("data.dat", "wb") as file:
                            pickle.dump(q_table, file)
                        return
                    if is_stalemate(board, 'black' if player_color == 'White' else 'white'):
                        print_board(board)
                        print("Stalemate! It's a draw.")
                        with open("data.dat", "wb") as file:
                            pickle.dump(q_table, file)
                        return
                    player_color = 'Black' if player_color == 'White' else 'White'
                else:
                    print("Invalid move. Try again.")
            else:
                print("Invalid move. Try again.")
    elif mode in ["computer", "play against the computer"]:
        print_board(board)
        first_open = True
        while True:
            move_valid = False
            while not move_valid:
                move = input("White's move (e.g., e2 e4): ").strip()
                if move.lower() == 'exit':
                    with open("data.dat", "wb") as file:
                        pickle.dump(q_table, file)
                    return
                parsed_move = parse_move(move)
                if parsed_move is None:
                    print("Invalid input, please enter a move in the correct format (e.g., e2 e4).")
                    continue
                start_pos, end_pos = parsed_move
                move_result = is_valid_move(board, start_pos, end_pos, "White", last_move)
                if move_result:
                    valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                    if valid_move:
                        handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                        print_board(board)
                        if is_checkmate(board, 'black'):
                            print("Checkmate! White wins!")
                            with open("data.dat", "wb") as file:
                                pickle.dump(q_table, file)
                            return
                        if is_stalemate(board, 'black'):
                            print("Stalemate! It's a draw.")
                            with open("data.dat", "wb") as file:
                                pickle.dump(q_table, file)
                            return
                        move_valid = True
                    else:
                        print("Invalid move, try again.")
                else:
                    print("Invalid move, try again.")
            if first_open:
                white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
                if white_move in openings:
                    computer_move = openings[white_move]
                    start_pos, end_pos = parse_move(computer_move)
                    valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                    print_board(board)
                    first_open = False
                    continue
            state = [list(row) for row in board]
            stime = time.time()
            action, _ = search(state)
            etime = time.time()
            ttime = etime - stime
            print(f"time: {ttime}")
            #action = ai_move(state)
            #action = choose_action(state)
            start_pos, end_pos = action
            valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
            if valid_move:
                handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                print_board(board)
                if is_checkmate(board, 'white'):
                    print("Checkmate! Black wins!")
                    with open("data.dat", "wb") as file:
                        pickle.dump(q_table, file)
                    return
                if is_stalemate(board, 'white'):
                    print("Stalemate! It's a draw.")
                    with open("data.dat", "wb") as file:
                        pickle.dump(q_table, file)
                    return
                reward = 0
                if is_check(board)["white"]:
                    reward = 1
                reward += evaluate(board) * -1
                next_state = [list(row) for row in board]
                update_q_table(state, action, reward, next_state)
                first_open = False
    else:
        print("Invalid choice.")
        play_chess()
def sim_chess():
    board = [row[:] for row in initial_board]
    last_move = None
    first_open = True
    print_board(board)
    while True:
        state = [list(row) for row in board]
        white_move, _ = search(board, 12, 0.5, "White")
        start_pos, end_pos = white_move
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        print_board(board)
        if valid_move:
            if is_checkmate(board, 'black'):
                print("Checkmate! White wins!")
                with open("data.dat", "wb") as file:
                    pickle.dump(q_table, file)
                return
            if is_stalemate(board, 'black'):
                print("Stalemate! It's a draw.")
                with open("data.dat", "wb") as file:
                    pickle.dump(q_table, file)
                return
            white_reward = 0
            if is_check(board)["black"]:
                white_reward = 1
            white_reward += evaluate(board)
            next_state = [list(row) for row in board]
            update_q_table(state, white_move, white_reward, next_state)
        state = [list(row) for row in board]
        black_move, _ = search(board, 10, 0.5, "Black")
        start_pos, end_pos = black_move
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        print_board(board)
        if valid_move:
            if is_checkmate(board, 'white'):
                print("Checkmate! Black wins!")
                with open("data.dat", "wb") as file:
                    pickle.dump(q_table, file)
                return
            if is_stalemate(board, 'white'):
                print("Stalemate! It's a draw.")
                with open("data.dat", "wb") as file:
                    pickle.dump(q_table, file)
                return
            reward = 0
            if is_check(board)["white"]:
                reward = 1
            reward += evaluate(board) * -1
            next_state = [list(row) for row in board]
            update_q_table(state, black_move, reward, next_state)
def main():
    '''
Also main game to play the chess'''
    print("This is chess.")
    print("If you do not know how to play chess, please consult someone who does.")
    instructions = input("Do you know how this program works? y/n ").strip().lower()
    if instructions != "y":
        print("You enter the starting algebraic coordinate and the ending algebraic coordinate in the input.")
        print("For promotion, you answer the question to promote.")
        print("q for queen, r for rook, n for knight, and b for bishop.")
    play_chess()
    play_again = input("Play again? y/n ").strip().lower()
    if play_again == "y":
        play_chess()
        print("Good game.")
        quit()
    else:
        print("Awww.")
        quit()
if __name__ == "__main__":
    main()
