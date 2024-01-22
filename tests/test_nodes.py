import torch
from marl.training.nodes import ValueNode, Add
from marl.models.batch import TransitionBatch, Batch
from .utils import MockEnv, generate_episode


def test_value_node():
    n = ValueNode("abc")
    assert n.value == "abc"
    n.value = "def"
    assert n.value == "def"


def test_add_node():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    res = Add(n1, n2)
    assert res.value == 65


def test_update_marks():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    res = Add(n1, n2)

    assert res._needs_update
    assert res.value == 65
    assert not res._needs_update


def test_update_marks_complex():
    """
    When updating n3, only res2 should be marked as needing an update.
    n1   n2   n3
     \\   /    |
      Add     |
     (res)    |
         \\    |
          \\   |
           \\  |
            Add
           (res2)
    """
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    n3 = ValueNode(10)
    res = Add(n1, n2)
    res2 = Add(res, n3)
    assert res2.value == 75

    assert not res._needs_update
    assert not res2._needs_update

    n3.value = -10
    assert not res._needs_update
    assert res2._needs_update
    assert res2.value == 55


def test_double_qlearning_node():
    """
    In Double q-learning, we should predict the action with the q-network and evaluate it with the q-target.
    As a result, we expect the values predicted by the DDQN node to come from the qtarget.

    In the case of this test, the qnetwork outputs [[0, 1, 2, 3, 4]] and the qtarget outputs [[10, 11, 12, 13, 14]].
    Therefore, the predicted values should be [[10, 11, 12, 13, 14]], indexed by the max action of [[0, 1, 2, 3, 4]],
    i.e. 4.
    """
    from marl.nn import LinearNN
    from marl.training.nodes import DoubleQLearning

    class MockNN(LinearNN):
        """Always returns the same qvalues for each agent: [1, 0, 0, 0, ..., 0]"""

        def __init__(self, output: torch.Tensor):
            super().__init__((1, 2, 3), (1, 2), output.shape)
            self.output = output

        def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
            return torch.tile(self.output, (obs.shape[0], 1, 1))

    env = MockEnv(1)

    # qnetwork outputs [[0, 1, 2, 3, 4]]
    # qtarget outputs [[10, 11, 12, 13, 14]]
    qnetwork = MockNN(torch.arange(0, env.n_actions, dtype=torch.float32).unsqueeze(0))
    qtarget = MockNN(torch.arange(0, env.n_actions, dtype=torch.float32).unsqueeze(0) + 10)

    episode = generate_episode(env)
    transitions = list(episode.transitions())
    batch = TransitionBatch(transitions, list(range(len(transitions))))
    batch_node = ValueNode[Batch](batch)  # type: ignore
    ddqn = DoubleQLearning(qnetwork, qtarget, batch_node)

    predicted = ddqn.value.squeeze()
    assert torch.all(predicted == qtarget.output[0, -1])