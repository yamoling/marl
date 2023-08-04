from marl.models import replay_memory


def test_memory_registry():
    memory = replay_memory.TransitionMemory(100)
    summary = memory.as_dict()
    reloaded = replay_memory.load(summary)
    assert isinstance(reloaded, replay_memory.TransitionMemory)
    assert reloaded.max_size == 100

    memory = replay_memory.EpisodeMemory(100)
    summary = memory.as_dict()
    reloaded = replay_memory.load(summary)
    assert isinstance(reloaded, replay_memory.EpisodeMemory)
    assert reloaded.max_size == 100

    memory = replay_memory.TransitionMemory(100)
    memory = replay_memory.PrioritizedMemory(memory)
    summary = memory.as_dict()
    reloaded = replay_memory.load(summary)
    assert isinstance(reloaded, replay_memory.PrioritizedMemory)
    assert isinstance(reloaded.memory, replay_memory.TransitionMemory)
    assert reloaded.max_size == 100
