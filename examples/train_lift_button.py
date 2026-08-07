from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks

MAP_PATH = "./lift3.toml"
# MAP_PATH = 2


def short_run():
    # `lift.toml` has 3 layers, so `obs_type="layered"` produces a 4D observation
    # shape that qnetworks.from_env doesn't support (only 1D/3D). Use "flattened".
    # `time_limit` otherwise defaults to `width * height // 2` == 4 steps, which
    # ignores the layers and is too short to reach the button, trigger it and ride
    # the lift.
    env = LLEConfig(MAP_PATH, obs_type="flattened", time_limit=32)
    trainer = algos.VDN(qnetworks.from_env(env))
    exp = Experiment.create(env, trainer, logdir="auto")
    # One run leaves the GPU at ~21% util: this 84k-param QMLP is launch-latency
    # bound, not compute bound. 8 concurrent seeds measured 1763 steps/s aggregate
    # (3.1x one run) at 99% util and <1GB VRAM. `n_jobs` must be explicit: "auto"
    # resolves to torch.cuda.device_count(), i.e. one run per GPU.
    exp.run(
        seeds=8,
        n_jobs=8,
        gpu_strategy="scatter",
        test_interval=10_000,
        n_tests=10,
        save_actions=False,
        quiet=True,
    )


if __name__ == "__main__":
    short_run()
