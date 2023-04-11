import marl


def test_get_test_metrics():
    run = marl.models.Run.load(
        "logs/test-replay-gamma.95-lvl2-new/run_1680944115.9170496"
    )
    episodes = run.get_test_episodes(5000)
