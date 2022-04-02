import numpy as np
import pygame
from controlAI import grid_conversion

#Controls
control_map = {
    #Uses the controls used in pygame
    'Left': [pygame.K_LEFT],
    'Right': [pygame.K_RIGHT],
    'Down': [pygame.K_DOWN],
    'Up': [pygame.K_UP], 
    'Space': [pygame.K_SPACE]
    }

feature_names = [
    'agg_height', 'n_holes', 'bumps', 'pits', 'max_wells', 'cols_w_holes', 'row_trans', 'col_trans'
    ]

#Needs to work with controlAI code to get piece info

def get_board_info(area) : # can use tetris (the game wrapper) and lines in the game
    
    peaks = get_peaks(area)
    highest_peak = np.max(peaks)

    agg_height = np.sum(peaks)

    holes = get_holes(peaks, area)
    n_holes = np.sum(holes)

    cols_w_holes = np.count_nonzero(np.array(holes) > 0)

    row_trans = get_row_trans(area, highest_peak)
    col_trans = get_col_trans(area, peaks)

    bumps = get_bumps(peaks)

    pits = np.count_nonzero(np.count_nonzero(area, axis = 0) == 0)

    wells = get_wells(peaks)
    max_wells = np.max(wells)

    return agg_height, n_holes, bumps, pits, max_wells, cols_w_holes, row_trans, col_trans

def get_peaks(area):
    #Peaks are the columns from top to bottom
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis = 0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    
    return peaks

def get_row_trans(area, highest_peak):
    sum = 0
    
    for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col] - 1:
                sum += 1
    
    return sum

def get_col_trans(area, peaks):
    sum =0 
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                sum += 1
    return sum

def get_bumps(peaks):
    s = 0
    for i in range(9):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s

def get_holes(peaks, area):
    holes = []
    for col in range(area.shape[1]):
        start = -peaks[col]
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(area[int(start):, col] == 0))

    return holes

def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) -1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    
    return wells

def check_turns(piece_letter):
    #How many turns a piece typically has
    if piece_letter == "I" or piece_letter == "S" or piece_letter == "Z":
        return 2
    
    if piece_letter == "O":
        return 1
    
    return 4

def check_needed_moves(piece_letter):
    #How many movies in general a piece would have
    if piece_letter == "S" or piece_letter == "Z":
        return 3, 5
    
    if piece_letter == "O":
        return 4, 4
    
    return 4, 5

def press_btn(action):
    #Simulates a keypress using pygame
    pygame.KEYDOWN
    control_map[action]

def make_move(action, tetris, n_dir, n_turn):
    for dir_count in range(1, n_dir + 1):
        for turn in range(1, n_turn + 1):

            for t in range(turn):
                press_btn('Up')
            
            if action != 'Middle':
                for move in range(dir_count):
                    press_btn(action)
            
            press_btn('Space')

            yield {'Turn': turn,
                   'Left': dir_count if action == 'Left' else 0,
                   'Right': dir_count if action == 'Right' else 0}


            