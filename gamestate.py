

import random
from enum import IntEnum, Enum
import numpy as np

Square = IntEnum('Square', 'EMPTY SNAKE')
#Turn = IntEnum('Turn', 'LEFT RIGHT STRAIGHT') # Used to change the current direction
class Turn():
    """ Used to change the current direction of the snake. """
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2

    @staticmethod
    def change_direction(dir_, turn):
        ccw = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]
        i = [d.value for d in ccw].index(dir_.value)
        
        if turn == Turn.STRAIGHT:
            return dir_
        elif turn == Turn.LEFT:
            return ccw[(i + 1) % 4]
        elif turn == Turn.RIGHT:
            return ccw[(i + 3) % 4]
        raise ValueError(f"Invalid Turn: {turn}")
            
    
class Direction(Enum):
    """ Direction of the snake. """
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)
    
    @staticmethod
    def from_vector(coord):
        if coord == (0, -1):
            return LEFT
        elif coord == (0, 1):
            return RIGHT
        elif coord == (-1, 0):
            return UP
        else:
            return DOWN
        raise ValueError(f"Invalid coordinate: {coord}")
        #vec_to_dir = { (0, -1) : LEFT, (0, 1) : RIGHT, (-1, 0) : UP, (1, 0) : DOWN }
        #return vec_to_dir[coord]
        
class GameState():

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.new_game()

    def new_game(self):
        # snake[i][j], inc i is the DOWN direction, inc j is the RIGHT direction.
        center = ((self.board_size-1)//2, (self.board_size-1)//2) 
        self.snake = [center] # Snake head is at index 0
        self.board = [[Square.EMPTY]*self.board_size for _ in range(self.board_size)]
        self.board[center[0]][center[1]] = Square.SNAKE
        self.direction = Direction.RIGHT
        self.score = 0
        self.has_ended = False
        self.food = None # index of food, we don't record food in board
        self.gen_food()

    def gen_food(self):
        num_empty = self.board_size**2 - len(self.snake)
        food_index = random.randrange(0, num_empty)

        sq = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] is Square.EMPTY:
                    if sq == food_index:
                        self.food = (i, j)
                        return
                    sq += 1

    def rotate_90(self):
        gs = self.copy()
        n = gs.board_size
        gs.snake = [(n-1-y, x) for x, y in gs.snake]
        for i in range(n):
            for j in range(n):
                gs.board[n-1-j][i] = self.board[i][j]
        gs.food = (n-1-self.food[1], self.food[0])
        rot = {Direction.RIGHT: Direction.UP, Direction.UP: Direction.LEFT,
               Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT}
        gs.direction = rot[self.direction]
        return gs

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def in_board(self, coord):
        for c in coord:
            if c < 0 or c >= self.board_size:
                return False
        return True

    def at(self, coord):
        return self.board[coord[0]][coord[1]]

    def make_move(self, turn):
        next_gs = self.copy()

        if next_gs.has_ended:
            return next_gs
        
        next_gs.direction = Turn.change_direction(self.direction, turn)
        new_head = tuple(map(sum, zip(self.snake[0], next_gs.direction.value)))
        
        if not self.in_board(new_head) or (self.at(new_head) is Square.SNAKE and new_head in self.snake[:-1]):
            next_gs.score -= 5.0
            next_gs.has_ended = True
            next_gs.food = None
            return next_gs

        next_gs.snake.insert(0, new_head)
        hd_x, hd_y = new_head
        next_gs.board[hd_x][hd_y] = Square.SNAKE
        
        if new_head != self.food:
            tail_x, tail_y = next_gs.snake[-1]
            if next_gs.snake[-1] != new_head:
                next_gs.board[tail_x][tail_y] = Square.EMPTY
            next_gs.snake = next_gs.snake[:-1]
            #next_gs.score -= 1.0/(2 * next_gs.board_size ** 2) # not getting the food costs
        else:
            # Got the food
            next_gs.score += 1.0
            if len(next_gs.snake) == next_gs.board_size ** 2: # win!
                next_gs.food = None
                next_gs.has_ended = True
            else:
                next_gs.gen_food()
            
        return next_gs

    def __str__(self):
        s = ''
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) == self.food:
                    s += 'F'
                elif self.board[i][j] is Square.EMPTY:
                    s += '-'
                else:
                    s += 'x'
            s += '\n'
        return s

    def plot(self, ax):
        ax.clear()
        ax.set_xticks(np.arange(0, self.board_size+1, 1.0))
        ax.set_yticks(np.arange(0, self.board_size+1, 1.0))
        ax.grid()
        ax.tick_params(labelbottom=False)
        ax.tick_params(labelleft=False)
        ax.set_xlim([0, self.board_size])
        ax.set_ylim([0, self.board_size])
        ax.set_aspect('equal')
        
        ys = [self.board_size - (coord[0] + 0.5) for coord in self.snake]
        xs = [coord[1] + 0.5 for coord in self.snake]
        ax.plot(xs, ys)

        dir_ = self.direction
        angle = 0
        if dir_ == Direction.RIGHT:
            angle = -90.
        elif dir_ == Direction.LEFT:
            angle = 90.
        elif dir_ == Direction.DOWN:
            angle = 180.
            
        ax.plot(xs[0] , ys[0], marker=(3, 0, angle), markersize=20, linestyle='None')
        #ax.plot(xs[:1], ys[:1], 'ro')
        
        if self.food is not None:
            y = [self.board_size - (self.food[0] + 0.5)]
            x = [self.food[1] + 0.5]
            ax.plot(x, y, 'bo')
        

print(Direction.LEFT.value)
print(Turn.change_direction(Direction.DOWN, Turn.RIGHT))
gs = GameState()
print(gs)




