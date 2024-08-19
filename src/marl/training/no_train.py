from .trainer import Trainer


class NoTrain(Trainer):
    def __init__(self):
        super().__init__("episode", 1)

    def to(self, _):
        return self

    def randomize(self):
        return
