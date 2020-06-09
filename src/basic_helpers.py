import math

from past.builtins import xrange

from venv.src.game_config import *


def cube_vertices(x, y, z, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n.
    """
    return [
        x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,  # bottom
        x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,  # left
        x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,  # right
        x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,  # front
        x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,  # back
    ]


def bomberman_vertices(x, y, z, vert_coods):
    adjusted_coords = []
    for idx, vertex in enumerate(vert_coods):
        '''
        if idx == 0:
            adjusted_coords.append(vertex + x)
        if idx == 1:
            adjusted_coords.append(vertex + y)
        if idx == 2:
            adjusted_coords.append(vertex + z)
        '''
        zbytek = idx % 3
        if zbytek == 0:
            adjusted_coords.append(vertex + x)
        if zbytek == 1:
            adjusted_coords.append(vertex + y)
        if zbytek == 2:
            adjusted_coords.append(vertex + z)
        '''
        adjusted_coords.append(float(vertex[0])+x)
        adjusted_coords.append(float(vertex[1])+y)
        adjusted_coords.append(float(vertex[2]) + z)
        '''
    print(adjusted_coords)
    print(str(len(adjusted_coords))+ 'here')
    return adjusted_coords


def get_int_from_float(float_nr):
    return int(math.modf(float_nr)[1])


def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.
    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m


def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side.
    """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result

def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.
    Parameters
    ----------
    position : tuple of len 3
    Returns
    -------
    block_position : tuple of ints of len 3
    """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return x, y, z


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.
    Parameters
    ----------
    position : tuple of len 3
    Returns
    -------
    sector : tuple of len 3
    """
    x, y, z = normalize(position)
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return x, 0, z


def is_starting_position(x, z, field_size):
    field_size = field_size / 2

    if x in (-field_size + 1, field_size - 1) and z in (-field_size + 1, field_size - 1):
        return True
    if x in (-field_size + 2, field_size - 1) and z in (-field_size + 1, field_size - 1):
        return True
    if x in (-field_size + 1, field_size - 2) and z in (-field_size + 1, field_size - 1):
        return True
    if x in (-field_size + 1, field_size - 1) and z in (-field_size + 2, field_size - 1):
        return True
    if x in (-field_size + 1, field_size - 1) and z in (-field_size + 1, field_size - 2):
        return True

    return False


def get_starting_positions(field_size):
    field_size /= 2
    field_size -= 1

    return [(field_size, -field_size), (-field_size, field_size), (-field_size, -field_size), (field_size, field_size)]


def pythagoras_get_c(a, b):
    return math.sqrt(a ^ 2 + b ^ 2)


def rotate(origin, point, angle):
    """
    Rotate a point by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def get_positions_list():
    n = HALF_OF_FIELD_SIZE
    s = 1  # step size
    y = 0  # initial y height

    result = []

    for x in xrange(-n, n + 1, s):
        for z in xrange(-n, n + 1, s):
            result.append((x, y, z))

    return result