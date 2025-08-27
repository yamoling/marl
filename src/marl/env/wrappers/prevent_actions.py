from dataclasses import dataclass
from lle import LLE, Action, World


@dataclass
class PreventActions(LLE):
    def __init__(self, width: int):
        super().__init__(World.from_file("maps/showcase"), name="PreventActions")
        self.b_width = width

    def available_actions(self):
        available = super().available_actions()
        i, j = self.world.agents_positions[0]
        # When right before the middle of the map
        if j == 1:
            if i == 0 and self.b_width < 4:
                available[0, Action.EAST.value] = False
            elif i == 1 and self.b_width < 2:
                available[0, Action.EAST.value] = False
            elif i == 2 and self.b_width < 1:
                available[0, Action.EAST.value] = False
            elif i == 3 and self.b_width < 3:
                available[0, Action.EAST.value] = False
            elif i == 4 and self.b_width < 5:
                available[0, Action.EAST.value] = False
        return available
