
def test_memory_registry():
    from marl.models import replay_memory
    memory = replay_memory.TransitionMemory(100)
    summary = memory.summary()
    reloaded = replay_memory.from_summary(summary)
    assert isinstance(reloaded, replay_memory.TransitionMemory)
    assert reloaded.max_size == 100

    memory = replay_memory.EpisodeMemory(100)
    summary = memory.summary()
    reloaded = replay_memory.from_summary(summary)
    assert isinstance(reloaded, replay_memory.EpisodeMemory)
    assert reloaded.max_size == 100

    memory = replay_memory.TransitionMemory(100)
    memory = replay_memory.PrioritizedMemory(memory)
    summary = memory.summary()
    reloaded = replay_memory.from_summary(summary)
    assert isinstance(reloaded, replay_memory.PrioritizedMemory)
    assert isinstance(reloaded._wrapped_memory, replay_memory.TransitionMemory)
    assert reloaded.max_size == 100