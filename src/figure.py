
from venv.src import game_field
from venv.src.basic_helpers import cube_vertices, get_int_from_float, bomberman_vertices
from venv.src.game_config import BOMB_STARTING_RANGE, BOMB_TIMESPAN_SECS
from venv.src.bomb import Bomb
from venv.src.ObjLoader import Model




class BaseFigure:
    def __init__(self, position_x, position_z):
        self.position_x = position_x
        self.position_z = position_z
        self.gl_object = None
        self.bomb_count = 1
        self.placed_bombs = 0
        self.hit = False
        self.previous_direction = None
        self.loader = Model()

    def recalculate_vertices(self):
        if self.gl_object is not None:
            self.gl_object.vertices = bomberman_vertices(self.position_x, 0, self.position_z,
                                                         self.loader.bomberman.vert_coords_dupli)

    def place_bomb(self):
        if self.placed_bombs < self.bomb_count:
            position_x = get_int_from_float(self.position_x)
            position_z = get_int_from_float(self.position_z)

            new_bomb = Bomb(self, position_x, position_z, BOMB_STARTING_RANGE, BOMB_TIMESPAN_SECS)

            self.placed_bombs += 1

            return new_bomb
        else:
            return None

    def remove_bomb(self, bomb):
        self.placed_bombs -= 1

    def get_normalized_positions(self, coef=0.25):
        x = self.position_x
        z = self.position_z

        if x > 0:
            x -= coef
        else:
            x += coef

        if z > 0:
            z -= coef
        else:
            z += coef

        return x, z