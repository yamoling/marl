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


def test_replace_parent():
    n1 = ValueNode(25)
    n2 = ValueNode(40)
    n3 = ValueNode(10)

    add1 = Add(n1, n2)
    assert add1.value == 65

    add1.replace_parent(n1, n3)
    # Check that the value has been updated
    assert add1.value == 50

    # Check that the parents and children have been updated
    assert len(add1.parents) == 2
    assert len(n1.children) == 0
    assert len(n3.children) == 1

    assert add1 not in n1.children
    assert add1 in n3.children

    assert n1 not in add1.parents
    assert n3 in add1.parents


def test_replace_node():
    n1 = ValueNode(25)
    n2 = ValueNode(40)

    res = Add(n1, n2)
    assert res.value == 65

    n3 = ValueNode(10)    
    n2.replace(n3)

    assert res.value == 35

    assert len(n2.children) == 0
    assert len(n3.children) == 1
    assert n3.children[0] == res

    # Check leaf parents
    assert len(res.parents) == 2 
    assert n3 in res.parents
    assert n1 in res.parents

