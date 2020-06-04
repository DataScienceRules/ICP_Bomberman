from venv.src.figure import BaseFigure


class PlayerFigure(BaseFigure):
    def __init__(self, position_x, position_z):
        super(self.__class__, self).__init__(position_x, position_z)