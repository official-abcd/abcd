#!/usr/bin/env python3
import sys
import time
import piece  # Import the piece module
import main

version = "abcd 2024"

hist = [None, [row[:] for row in piece.initial_board]]  # Initialize history with initial board

while True:
    input_line = sys.stdin.readline()
    if not input_line:
        break
    input_line = input_line.strip()
    if not input_line:
        continue
    args = input_line.split()
    if args[0] == "uci":
        print(f"id name {version}")
        print("uciok")
    elif args[0] == "isready":
        print("readyok")
    elif args[0] == "quit":
        break
    elif args[0] == "position":
        # Handle 'position' command
        if args[1] == "startpos":
            board = [row[:] for row in piece.initial_board]
            moves = []
            if len(args) > 2 and args[2] == "moves":
                moves = args[3:]
        elif args[1] == "fen":
            # Handle FEN position
            fen_parts = args[2:]
            if "moves" in fen_parts:
                moves_index = fen_parts.index("moves")
                fen_str = ' '.join(fen_parts[:moves_index])
                moves = fen_parts[moves_index+1:]
            else:
                fen_str = ' '.join(fen_parts)
                moves = []
            board = piece.parse_fen(fen_str)
        else:
            continue  # Invalid 'position' command
        hist = [None, board]
        last_move = None
        for move_str in moves:
            move = piece.parse_uci_move(move_str)
            if move is None:
                #print(f"Invalid move: {move_str}")
                continue
            board = [row[:] for row in hist[-1]]
            valid_move, last_move = piece.move_piece(board, move[0], move[1], last_move)
            if valid_move:
                hist.append(board)
            else:
                #print(f"Illegal move: {move_str}")
                break
    elif args[0] == "go":
        # Handle 'go' command
        wtime = btime = winc = binc = None
        # Parse time controls if provided
        for i in range(1, len(args)-1):
            if args[i] == 'wtime':
                wtime = int(args[i+1])
            elif args[i] == 'btime':
                btime = int(args[i+1])
            elif args[i] == 'winc':
                winc = int(args[i+1])
            elif args[i] == 'binc':
                binc = int(args[i+1])
        # Determine time to think
        if len(hist) % 2 == 0:
            # Black to move
            time_left = btime or 1000  # Default to 1 sec if not provided
            increment = binc or 0
        else:
            # White to move
            time_left = wtime or 1000
            increment = winc or 0
        think_time = min(time_left / 40 + increment, time_left / 2 - 1) / 1000  # Convert to seconds
        move, depth_reached = piece.search(hist[-1], 18, think_time)
        if move:
            move_str = main.convert_move_to_uci(move, hist[-1])
            print(f"bestmove {move_str}")
            # Update history
            board = [row[:] for row in hist[-1]]
            valid_move, last_move = piece.move_piece(board, move[0], move[1], last_move)
            if valid_move:
                hist.append(board)
            else:
                # more stuff here.
                pass
        else:
            print("bestmove 0000")
    elif args[0] == "stop":
        # Stop calculating and output best move
        pass  # Implement stop command if needed
    else:
        # Handle other commands
        pass
