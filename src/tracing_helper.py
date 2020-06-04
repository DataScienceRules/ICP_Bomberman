import math
from collections import namedtuple, deque
from venv.src.basic_helpers import get_positions_list
from venv.src.game_config import TRACING_GRASS_CONSTANT, HALF_OF_FIELD_SIZE
from venv.src.textures import *

Edge = namedtuple('Edge', 'start_z, start_x, end_z, end_x, cost')


def make_edge(start_z, start_x, end_z, end_x, cost=1):
    return Edge(start_z, start_x, end_z, end_x, cost)


class TracingHelper:
    def __init__(self, game_field):
        self.game_field = game_field
        self.quantified_game_field = None
        self._neighbours = None
        self.vertices = get_positions_list()

    def quantify_game_field(self):
        self.quantified_game_field = []

        for position in self.vertices:
            x, y, z = position

            dx = x - 1
            if math.fabs(dx) <= HALF_OF_FIELD_SIZE:
                block = self.game_field.shown.get((dx, 0, z))
                if block is not None:
                    if block == GRASS:
                        self.quantified_game_field.append(make_edge(z, x, z, dx, TRACING_GRASS_CONSTANT))
                else:
                    self.quantified_game_field.append(make_edge(z, x, z, dx))

            dx = x + 1
            if math.fabs(dx) <= HALF_OF_FIELD_SIZE:
                block = self.game_field.shown.get((dx, 0, z))
                if block is not None:
                    if block == GRASS:
                        self.quantified_game_field.append(make_edge(z, x, z, dx, TRACING_GRASS_CONSTANT))
                else:
                    self.quantified_game_field.append(make_edge(z, x, z, dx))

            dz = z - 1
            if math.fabs(dz) <= HALF_OF_FIELD_SIZE:
                block = self.game_field.shown.get((x, 0, dz))
                if block is not None:
                    if block == GRASS:
                        self.quantified_game_field.append(make_edge(z, x, dz, x, TRACING_GRASS_CONSTANT))
                else:
                    self.quantified_game_field.append(make_edge(z, x, dz, x))

            dz = z + 1
            if math.fabs(dz) <= HALF_OF_FIELD_SIZE:
                block = self.game_field.shown.get((x, 0, dz))
                if block is not None:
                    if block == GRASS:
                        self.quantified_game_field.append(make_edge(z, x, dz, x, TRACING_GRASS_CONSTANT))
                else:
                    self.quantified_game_field.append(make_edge(z, x, dz, x))

        pass

    @property
    def is_quantified(self):
        return self.quantified_game_field is not None

    @property
    def neighbours(self):
        if self._neighbours is None:
            self._neighbours = {vertex: set() for vertex in self.vertices}

            for edge in self.quantified_game_field:
                self._neighbours[(edge.start_x, 0, edge.start_z)].add(((edge.end_x, 0, edge.end_z), edge.cost))

        return self._neighbours

    def do_dijkstra(self, start, destination):
        if self.is_quantified is False:
            self.quantify_game_field()

        x, y, z = start
        start = round(x), round(y), round(z)

        x, y, z = destination
        destination = round(x), round(y), round(z)

        assert start in self.vertices, 'source node not exists'

        distances = {vertex: math.inf for vertex in self.vertices}

        previous_vertices = {
            vertex: None for vertex in self.vertices
        }

        distances[start] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])

            vertices.remove(current_vertex)

            if distances[current_vertex] == math.inf:
                break

            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost

                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), destination

        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]

        if path:
            path.appendleft(current_vertex)

        return path

    def get_direction_from_path(self, path):
        if len(list(path)) < 2:
            return 0, 0

        start_x, start_y, start_z = path[0]
        next_x, next_y, next_z = path[1]

        return next_x - start_x, next_z - start_z