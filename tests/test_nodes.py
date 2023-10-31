from marl.training.nodes import ValueNode, Add


def test_value_node():
    n = ValueNode("abc")
    assert n.value == "abc"
    n.value = 25
    assert n.value == 25


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
     \   /    |
      Add     |
     (res)    |
         \    |  
          \   |
           \  |
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

