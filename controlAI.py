from turtle import pos, shape
import pygame

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]


def grid_conversion(grid):

    new_grid = []

    new_grid = grid


    return new_grid
    

def fig_conversion(current_piece):
    fig = []

    if current_piece == ['..0..' , '..0..' , '..0..' , '..0..' , '.....']:
        fig = [1, 5, 9, 13]

    if current_piece == ['.....' , '0000.' , '.....' , '.....' , '.....']:
        fig = [4, 5, 6, 7]

    if current_piece == ['.....' , '.....' , '..00.',  '.00..', '.....']:
        fig = [5, 6, 8, 9]

    if current_piece == ['.....' , '..0..' , '..00.' , '...0.' , '.....']:
      fig = [1, 5, 6, 10]
        
    if current_piece == ['.....' , '.....' , '.00..' , '..00.' , '.....']:
        fig = [5, 6, 10, 11]

    if current_piece == ['.....', '..0..', '.00..', '.0...', '.....']:
      fig = [1, 5, 4, 8]

    if current_piece == ['.....' , '.....' , '.00..' , '.00..' , '.....']:
        fig = [1, 2, 5, 6]
        
    if current_piece == ['.....', '..00.', '..0..', '..0..', '.....']:
        fig = [1, 2, 5, 9] 
        
    if current_piece == ['.....', '.0...', '.000.', '.....', '.....']:
        fig = [0, 4, 5, 6] 
            
    if current_piece == ['.....', '..0..', '..0..', '.00..', '.....']:
        fig = [1, 5, 9, 8] 
                
    if current_piece == ['.....', '.....', '.000.', '...0.', '.....']:
        fig = [4, 5, 6, 10] 

    if current_piece == ['.....', '.00..', '..0..', '..0..', '.....']:
        fig = [1, 2, 6, 10]   

    if current_piece == ['.....', '.....', '.000.', '.0...', '.....']:
        fig = [5, 6, 7, 9]

    if current_piece == ['.....', '..0..', '..0..', '..00.', '.....']:
        fig = [2, 6, 10, 11]

    if current_piece == ['.....', '...0.', '.000.', '.....', '.....']:
        fig = [3, 5, 6, 7] 

    if current_piece == ['.....', '..0..', '.000.', '.....', '.....']:
        fig = [1, 4, 5, 6]

    if current_piece == ['.....', '..0..', '.00..', '..0..', '.....']:
        fig = [1, 4, 5, 9]

    if current_piece == ['.....', '.....', '.000.', '..0..', '.....']:
        fig = [4, 5, 6, 9]

    if current_piece == ['.....', '..0..' , '..00.' , '..0..' , '.....']:
        fig = [1, 5, 6, 9]

    return fig



class Event():
    type = None
    key = None

    def __init__(self, type, key):
        self.type = type
        self.key = key

counter = 0

# Basic function to make the AI rotate endlessly, at the moment is useful to test if the two files work well together.
def run_ai(grid, play_width, play_height, current_piece):
    global counter
    counter += 1
    piece_fig = fig_conversion(current_piece)
    # Depending on your PC this counter rate may need to be adjusted. More if system is faster and vice versa 
    if counter < 3:
        return []
    counter = 0
    rotation, position = best_rot_pos(grid, play_width, play_height, current_piece, piece_fig)
    if current_piece.rotation != rotation:
        e = Event(pygame.KEYDOWN, pygame.K_UP)
    elif current_piece.x < position:
        e = Event(pygame.KEYDOWN, pygame.K_RIGHT)
    elif current_piece.x > position:
        e = Event(pygame.KEYDOWN, pygame.K_LEFT)
    else:
        e = Event(pygame.KEYDOWN, pygame.K_SPACE)
    return [e]


# parameters and arguments might need changing in the future for this method, keep an eye on them if any issues occur.
# Idea of this function is to see if the current row on the board has any holes in it for a piece to fit into.
def will_it_fit(grid, x, y, play_width, play_height, current_piece, piece_fig):

    will_it_fit = False
    current_piece = fig_conversion(current_piece)
    new_grid = grid_conversion(grid)
    # Check in a 4 by 4 square (i and j) to see if it is out of bounds or not
    for i in range(4):
        for j in range(4):
            if i * 4 + j in piece_fig:
                if i + y > play_height - 1 or \
                   j + x > play_width - 1 or \
                   j + x < 0 or \
                   new_grid[i + y][j + x] > 0:
                       will_it_fit = True



    return will_it_fit



# simulate is the method that checks rotation and position of the free squares to check if it can be placed there
# Simulate tries to see if any rotation will fit in the current bricks
def simulate(grid, x, y, play_width, play_height, current_piece, piece_fig):
    # NOTE: While bottom row doesn't have a match, it will recursively check the rows moving up
    current_piece = fig_conversion(current_piece)
    while not will_it_fit(grid, x, y, play_width, play_height, current_piece, piece_fig):
        y += 1
        print("it doesn't fit on this line")
    y-= 1

    height = play_height
    holes = 0
    filled = []
    for i in range(play_height -1, -1, -1,):
        it_is_full = True
        prev_holes = holes
        for j in range (play_width):
            # if u ever becomes "x" during these searches, it means the hole it is checking is empty and it will be appended later in the method
            u = '_'
            # this if just checks outright if that position is empty
            if grid[i][j] != 0:
                u = "x"
            # this check is for the 4x4 squares around the main position
            for ii in range(4):
                for jj in range(4):
                    if ii * 4 + jj in current_piece:
                        if jj + x == j and ii + y == i:
                            u = "x"
            # here we check to see if the ai is checking the bottom of the grid or did it go higher
            if u == "x" and i < height:
                # if it's higher we update it.
                height = i
                if u == "x":
                    # we append the position of the squares that are empty
                    filled.append((i, j))
                    for k in range(i, play_height):
                        if (k, j) not in filled:
                            holes += 1
                            filled.append((k, j))
                else:
                    it_is_full = False
            if it_is_full:
                breaks += 1
                holes = prev_holes

    return holes, play_height - height

# Focusing on simulating each position 
def best_rot_pos(grid, play_width, play_height, current_piece, piece_fig):
    best_pos = None
    best_rot = None
    best_height = play_height
    best_holes = play_height * play_width
    height = best_height


    for rotation in range(len(current_piece.shape)):
        fig_shape = current_piece.shape[rotation]
        for j in range (-3, play_width):
            if not will_it_fit(grid, j, 0, play_width, play_height, fig_shape, piece_fig):
                holes, height = simulate(grid, j, 0, play_width, play_height, fig_shape, piece_fig)
            if best_pos is None or best_holes > holes or \
                best_holes == holes and best_height > height:
                best_height = height
                best_holes = holes
                best_pos = j
                best_rot = rotation
    return best_rot, best_pos



#This is where the AI will control the Tetris game from

#TODO: Access the current status of the board
#TODO: Based on current block assess best placement
