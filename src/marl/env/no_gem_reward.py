from lle import LLE


class NoGemLLE(LLE):
    def __init__(self, args):
        super().__init__(args)

    def reset(self):
        obs = super().reset()
