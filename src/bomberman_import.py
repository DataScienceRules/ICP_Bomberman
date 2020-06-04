from pyglet.gl import *
import ctypes
import pyrr
import time
from ObjLoader import ObjLoader

class Model:
    def __init__(self):

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
        glBufferData(GL_ARRAY_BUFFER, len(self.tree.model)*4, self.tree.c_model, GL_STATIC_DRAW)

        texture_offset = len(self.tree.vertex_index) * 12
        normal_offset = (texture_offset + len(self.tree.texture_index) * 8)

        #positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        #textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 2, ctypes.c_void_p(texture_offset))
        glEnableVertexAttribArray(1)

        #normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.tree.model.itemsize * 3, ctypes.c_void_p(normal_offset))
        glEnableVertexAttribArray(2)

        texture = GLuint(0)
        glGenTextures(1, texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        #set the texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        #set the texture filtering
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

    def rotate(self):

        ct = time.clock()
        self.rot_y = pyrr.Matrix44.from_y_rotation(ct).flatten()

        c_rotate = (GLfloat * len(self.rot_y))(*self.rot_y)

        glUniformMatrix4fv(self.rotate_loc, 1, GL_FALSE, c_rotate)
        glUniformMatrix4fv(self.light_loc, 1, GL_FALSE, c_rotate)