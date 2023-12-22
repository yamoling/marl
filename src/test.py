from lle import World


def extract(world: World):
    state = world.get_state()
    res = list(*state.agents_positions) + state.gems_collected
    for agent_pos in world.agents_positions:
        for pos, gem in world.gems:
            res.append(pos[0] - agent_pos[0])
            res.append(pos[1] - agent_pos[1])
        for exit_pos in world.exit_pos:
            res.append(exit_pos[0] - agent_pos[0])
            res.append(exit_pos[1] - agent_pos[1])
