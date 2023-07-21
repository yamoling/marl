import torch

from marl.qlearning.trainer.node import ValueNode, AddNode


def test_value_node():
    n = ValueNode("abc")
    assert n.value == "abc"
    n.value = 25
    assert n.value == 25


def test_add_node():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    res = AddNode(n1, n2)
    assert res.value == 65


def test_loss_node_zero():
    n1 = ValueNode(torch.Tensor([1, 2, 3]))
    n2 = ValueNode(torch.Tensor([1, 2, 3]))
    n3 = LossNode(n1, n2, torch.nn.MSELoss())
    loss = n3.value
    assert loss.item() == 0.


def test_loss_node_non_zero():
    n1 = ValueNode(torch.Tensor([1, 2, 3]))
    n2 = ValueNode(torch.Tensor([2, 2, 3]))
    n3 = LossNode(n1, n2, torch.nn.MSELoss())
    loss = n3.value.item()
    assert abs(loss - 1/3) < 1e-6


def test_replace_parent():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    n3 = ValueNode(10)

    add1 = AddNode(n1, n2)
    assert add1.value == 65

    add1.replace_parent(n1, n3)
    assert add1.value == 50


def test_replace_node():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    n3 = ValueNode(10)

    add1 = AddNode(n1, n2)
    assert add1.value == 65

    n2.replace_by(n3)
    assert add1.value == 35
