import time
import ctypes
import pyrr

from pyglet import *
from pyglet.gl import *
from pyglet.graphics import TextureGroup

from venv.src.basic_helpers import *
from venv.src.game_config import *
from venv.src.npc_figure import NPCFigure
from venv.src.player_figure import PlayerFigure
from venv.src.textures import *
from collections import deque

from venv.src.tracing_helper import TracingHelper
#from venv.src.bomberman_import import Model
from venv.src.ObjLoader import Model, ObjLoader


class GameField(object):

    def __init__(self):

        # A Batch is a collection of vertex lists for batched rendering.
        self.main_batch = pyglet.graphics.Batch()
        self.bomb_batch = pyglet.graphics.Batch()

        # A TextureGroup manages an OpenGL texture.
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())

        # A mapping from position to the texture of the block at that position.
        # This defines all the blocks that are currently in the world.
        self.world = {}

        # Same mapping as `world` but only contains blocks that are shown.
        self.shown = {}

        # Mapping from position to a pyglet `VertextList` for all shown blocks.
        self._shown = {}

        # Mapping from sector to a list of positions inside that sector.
        self.sectors = {}

        # Simple function queue implementation. The queue is populated with
        # _show_block() and _hide_block() calls
        self.queue = deque()

        self.bombs = deque([])

        self.tracing_helper = TracingHelper(self)

        '''
        self.tree = ObjLoader()
        self.tree.load_model("C:/Users/Krystof/PycharmProjects/Bomberman/venv/bomberman.obj")

        self.vertex_shader_source = b"""
                #version 330
                in layout(location = 0) vec3 positions;
                in layout(location = 1) vec2 textureCoords;
                in layout(location = 2) vec3 normals;
                uniform mat4 light;
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 rotate;
                out vec2 textures;
                out vec3 fragNormal;
                void main()
                {
                    fragNormal = (light * vec4(normals, 0.0f)).xyz;
                    gl_Position =  projection * view * model * rotate * vec4(positions, 1.0f);
                    textures = vec2(textureCoords.x, 1 - textureCoords.y);
                }
                """

        self.fragment_shader_source = b"""
                #version 330
                in vec2 textures;
                in vec3 fragNormal;
                uniform sampler2D sampTexture;
                out vec4 outColor;
                void main()
                {
                    vec3 ambientLightIntensity = vec3(0.3f, 0.2f, 0.4f);
                    vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
                    vec3 sunLightDirection = normalize(vec3(1.0f, 1.0f, -0.5f));
                    vec4 texel = texture(sampTexture, textures);
                    vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
                    outColor = vec4(texel.rgb * lightIntensity, texel.a);
                }
                """

        vertex_buff = ctypes.create_string_buffer(self.vertex_shader_source)
        c_vertex = ctypes.cast(ctypes.pointer(ctypes.pointer(vertex_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, 1, c_vertex, None)
        glCompileShader(vertex_shader)

        fragment_buff = ctypes.create_string_buffer(self.fragment_shader_source)
        c_fragment = ctypes.cast(ctypes.pointer(ctypes.pointer(fragment_buff)), ctypes.POINTER(ctypes.POINTER(GLchar)))
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, 1, c_fragment, None)
        glCompileShader(fragment_shader)

        shader = glCreateProgram()
        glAttachShader(shader, vertex_shader)
        glAttachShader(shader, fragment_shader)
        glLinkProgram(shader)

        glUseProgram(shader)

        vbo = GLuint(0)
        glGenBuffers(1, vbo)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, len(self.tree.model) * 4, self.tree.c_model, GL_STATIC_DRAW)

        texture_offset = len(self.tree.vertex_index) * 12
        normal_offset = (texture_offset + len(self.tree.texture_index) * 8)

        # positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 2, ctypes.c_void_p(texture_offset))
        glEnableVertexAttribArray(1)

        # normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(normal_offset))
        glEnableVertexAttribArray(2)

        texture = GLuint(0)
        glGenTextures(1, texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        # set the texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set the texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        xmas = pyglet.image.load('C:\\Users\Krystof\PycharmProjects\Bomberman\\venv\\bomberman.jpg')
        image_data = xmas.get_data('RGB', xmas.pitch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xmas.width, xmas.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

        view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0])).flatten()
        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1280 / 720, 0.1, 100.0).flatten()
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.5, -2.0])).flatten()

        c_view = (GLfloat * len(view))(*view)
        c_projection = (GLfloat * len(projection))(*projection)
        c_model = (GLfloat * len(model))(*model)

        self.rotate_loc = glGetUniformLocation(shader, b'rotate')
        self.view_loc = glGetUniformLocation(shader, b"view")
        self.proj_loc = glGetUniformLocation(shader, b"projection")
        self.model_loc = glGetUniformLocation(shader, b"model")
        self.light_loc = glGetUniformLocation(shader, b"light")

        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, c_view)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, c_projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, c_model)

        self.rot_y = pyrr.Matrix44.identity()

        '''
        self.player_figure, self.npc_figures = self._initialize_figures()

        self._initialize()

    def _initialize_figures(self):
        starting_positions = get_starting_positions(HALF_OF_FIELD_SIZE * 2)

        player_figure = PlayerFigure(starting_positions[0][0], starting_positions[0][1])
        npc_figure_one = NPCFigure(starting_positions[1][0], starting_positions[1][1])

        return player_figure, [npc_figure_one]

    def _initialize(self):
        """ Initialize the world by placing all the blocks.
        """
        n = HALF_OF_FIELD_SIZE
        s = 1  # step size
        y = 0  # initial y height

        for x in xrange(-n, n + 1, s):
            for z in xrange(-n, n + 1, s):

                if is_starting_position(x, z, n * 2) is False:
                    if (x % 2) == 0 or (z % 2) == 0:
                        self.add_block((x, y, z), GRASS, immediate=False)
                    else:
                        self.add_block((x, y, z), STONE, immediate=False)

                self.add_block((x, y - 1, z), STONE, immediate=False)

                if x in (-n, n) or z in (-n, n):
                    # create outer walls.
                    for dy in xrange(0, 1):
                        self.add_block((x, y + dy, z), STONE, immediate=False)

        self.show_figures(([self.player_figure] + self.npc_figures))

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.
        """
        x, y, z = position

        for dx, dy, dz in FACES:

            if (x + dx, y + dy, z + dz) not in self.world:
                return True

        return False

    def show_figures(self, figures):



        for figure in figures:
            x, z = figure.position_x, figure.position_z
            self.figure_model = Model()
            vert_coords = self.figure_model.vertex_data
            vertex_data_adjusted = bomberman_vertices(x, 0, z, vert_coords)




            #x, z = figure.position_x, figure.position_z
            #vertex_data_adjusted = bomberman_vertices(x, 0, z, vertex_data)

            print(len(vertex_data_adjusted))
            #print(num_verts)
            colors = []
            for one in vertex_data_adjusted:
                colors.append(1.0)
            print(len(colors))
            figure.gl_object = self.main_batch.add(1548, GL_TRIANGLES, None, ('v3f/dynamic', vertex_data_adjusted),
                                                   ('c3f', colors))



    '''
            ('t2f/static', texture_data_flat), ('n3f/dynamic', vertex_normals_flat)                                       
            
            vertex_data = cube_vertices(x, 0, z, 0.25)
            

            # create vertex list
            figure.gl_object = self.main_batch.add(24, GL_QUADS, self.group, ('v3f/dynamic', vertex_data),
                                                   ('t2f/static', texture_data))
            self.main_batch()
        '''
        #texture_data = list(BRICK)



            # create vertex list 24,  self.group

            #self.main_batch.add()

            
            #figure.model = Model()


    def add_block(self, position, texture, immediate=True):
        """ Add a block with the given `texture` and `position` to the world.
        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        immediate : bool
            Whether or not to draw the block immediately.
        """
        if position in self.world:
            self.remove_block(position, immediate)

        self.world[position] = texture
        self.sectors.setdefault(sectorize(position), []).append(position)

        if immediate:
            if self.exposed(position):
                self.show_block(position)

            self.check_neighbors(position)

    def remove_block(self, position, immediate=True):
        """ Remove the block at the given `position`.
        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to remove.
        immediate : bool
            Whether or not to immediately remove block from canvas.
        """
        del self.world[position]
        self.sectors[sectorize(position)].remove(position)

        if immediate:
            if position in self.shown:
                self.hide_block(position)

            self.check_neighbors(position)

    def check_neighbors(self, position):
        """ Check all blocks surrounding `position` and ensure their visual
        state is current. This means hiding blocks that are not exposed and
        ensuring that all exposed blocks are shown. Usually used after a block
        is added or removed.
        """
        x, y, z = position

        for dx, dy, dz in FACES:

            key = (x + dx, y + dy, z + dz)

            if key not in self.world:
                continue
            if self.exposed(key):
                if key not in self.shown:
                    self.show_block(key)
            else:
                if key in self.shown:
                    self.hide_block(key)

    def show_block(self, position, immediate=True):
        """ Show the block at the given `position`. This method assumes the
        block has already been added with add_block()
        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.
        immediate : bool
            Whether or not to show the block immediately.
        """
        texture = self.world[position]
        self.shown[position] = texture

        if immediate:
            self._show_block(position, texture)
        else:
            self._enqueue(self._show_block, position, texture)

    def _show_block(self, position, texture):
        """ Private implementation of the `show_block()` method.
        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        """
        x, y, z = position
        vertex_data = cube_vertices(x, y, z, 0.5)
        texture_data = list(texture)

        # create vertex list
        self._shown[position] = self.main_batch.add(24, GL_QUADS, self.group, ('v3f/static', vertex_data),
                                                    ('t2f/static', texture_data))

    def hide_block(self, position, immediate=True):
        """ Hide the block at the given `position`. Hiding does not remove the
        block from the world.
        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to hide.
        immediate : bool
            Whether or not to immediately remove the block from the canvas.
        """
        self.shown.pop(position)

        if immediate:
            self._hide_block(position)
        else:
            self._enqueue(self._hide_block, position)

    def _hide_block(self, position):
        """ Private implementation of the 'hide_block()` method.
        """
        self._shown.pop(position).delete()

    def check_if_hit_npcs(self, x, z):
        hit_npcs = []

        for npc in self.npc_figures:
            if round(npc.position_x) == x and round(npc.position_z) == z:
                hit_npcs.append(npc)

        return hit_npcs

    def check_if_hit_player(self, x, z):
        return get_int_from_float(self.player_figure.position_x) == x \
               and get_int_from_float(self.player_figure.position_z) == z

    def remove_figures(self, figures):
        for figure in figures:
            self.remove_figure(figure)

    def remove_figure(self, figure):
        # figure.gl_object.delete()
        figure.hit = True

    def detonation(self, dt):
        if len(self.bombs) > 0:
            bomb = self.bombs.popleft()
            _range = bomb.range

            for dx in range(bomb.position_x, (bomb.position_x - _range), -1):
                block = self.shown.get((dx, 0, bomb.position_z))
                if block is not None:
                    if block == GRASS:
                        self.hide_block((dx, 0, bomb.position_z))
                    elif block == STONE:
                        break



                hit_npcs = self.check_if_hit_npcs(dx, bomb.position_z)
                self.remove_figures(hit_npcs)

                if self.check_if_hit_player(dx, bomb.position_z):
                    self.player_figure.hit = True

            for dx in range(bomb.position_x, (bomb.position_x + _range), 1):
                block = self.shown.get((dx, 0, bomb.position_z))
                if block is not None:
                    if block == GRASS:
                        self.hide_block((dx, 0, bomb.position_z))
                    elif block == STONE:
                        break

                hit_npcs = self.check_if_hit_npcs(dx, bomb.position_z)
                self.remove_figures(hit_npcs)

                if self.check_if_hit_player(dx, bomb.position_z):
                    self.player_figure.hit = True

            for dz in range(bomb.position_z, bomb.position_z - _range, -1):
                block = self.shown.get((bomb.position_x, 0, dz))
                if block is not None:
                    if block == GRASS:
                        self.hide_block((bomb.position_x, 0, dz))
                    elif block == STONE:
                        break

                hit_npcs = self.check_if_hit_npcs(bomb.position_x, dz)
                self.remove_figures(hit_npcs)

                if self.check_if_hit_player(bomb.position_x, dz):
                    self.player_figure.hit = True

            for dz in range(bomb.position_z, bomb.position_z + _range, 1):
                block = self.shown.get((bomb.position_x, 0, dz))
                if block is not None:
                    if block == GRASS:
                        self.hide_block((bomb.position_x, 0, dz))
                    elif block == STONE:
                        break

                hit_npcs = self.check_if_hit_npcs(bomb.position_x, dz)
                self.remove_figures(hit_npcs)

                if self.check_if_hit_player(bomb.position_x, dz):
                    self.player_figure.hit = True

            bomb.figure.placed_bombs -= 1
            bomb.figure.escaping_to = None

            self.tracing_helper = TracingHelper(self)

            del bomb

    def draw_bomb(self, bomb):
        x, y, z = bomb.position_x, 1, bomb.position_z
        vertex_data = cube_vertices(x, y, z, 0.5)
        texture_data = list(STONE)

        return pyglet.graphics.draw(24, GL_QUADS, ('v3f/static', vertex_data), ('t2f/static', texture_data))

    def show_sector(self, sector):
        """ Ensure all blocks in the given sector that should be shown are
        drawn to the canvas.
        """
        for position in self.sectors.get(sector, []):

            if position not in self.shown and self.exposed(position):
                self.show_block(position, False)

    def hide_sector(self, sector):
        """ Ensure all blocks in the given sector that should be hidden are
        removed from the canvas.
        """
        for position in self.sectors.get(sector, []):

            if position in self.shown:
                self.hide_block(position, False)

    def change_sectors(self, before, after):
        """ Move from sector `before` to sector `after`. A sector is a
        contiguous x, y sub-region of world. Sectors are used to speed up
        world rendering.
        """
        before_set = set()
        after_set = set()
        pad = 4

        for dx in xrange(-pad, pad + 1):

            for dy in [0]:

                for dz in xrange(-pad, pad + 1):

                    if dx ** 2 + dy ** 2 + dz ** 2 > (pad + 1) ** 2:
                        continue
                    if before:
                        x, y, z = before
                        before_set.add((x + dx, y + dy, z + dz))
                    if after:
                        x, y, z = after
                        after_set.add((x + dx, y + dy, z + dz))

        show = after_set - before_set
        hide = before_set - after_set

        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

    def _enqueue(self, func, *args):
        """ Add `func` to the internal queue.
        """
        self.queue.append((func, args))

    def _dequeue(self):
        """ Pop the top function from the internal queue and call it.
        """
        func, args = self.queue.popleft()
        func(*args)

    def process_queue(self):
        """ Process the entire queue while taking periodic breaks. This allows
        the game loop to run smoothly. The queue contains calls to
        _show_block() and _hide_block() so this method should be called if
        add_block() or remove_block() was called with immediate=False
        """
        start = time.clock()

        while self.queue and time.clock() - start < 1.0 / TICKS_PER_SEC:
            self._dequeue()

    def process_entire_queue(self):
        """ Process the entire queue with no breaks.
        """
        while self.queue:
            self._dequeue()

    def check_if_figure_collide(self, position_x, position_z):
        figure_size_half = 0.25

        position_x_left = round(position_x + figure_size_half)
        position_x_right = round(position_x - figure_size_half)
        position_z_top = round(position_z + figure_size_half)
        position_z_bottom = round(position_z - figure_size_half)

        x = get_int_from_float(position_x)
        y = 0
        z = get_int_from_float(position_z)

        borders = HALF_OF_FIELD_SIZE - 1

        if math.fabs(position_x) - 0.25 > borders or math.fabs(position_z) - 0.25 > borders:
            return True
        else:
            left_collide = self._shown.get((position_x_left, 0, get_int_from_float(z))) is not None
            right_collide = self._shown.get(
                (position_x_right, 0, get_int_from_float(z))) is not None
            top_collide = self._shown.get((get_int_from_float(x), 0, position_z_top)) is not None
            bottom_collide = self._shown.get(
                (get_int_from_float(x), 0, position_z_bottom)) is not None

            return left_collide or right_collide or top_collide or bottom_collide