from venv.src.figure import BaseFigure


class NPCFigure(BaseFigure):
    def __init__(self, position_x, position_z):
        super(self.__class__, self).__init__(position_x, position_z)
        self.escaping_to = None