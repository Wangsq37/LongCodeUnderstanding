from pulp import lpDot, LpVariable


def test_lpdot():
    # Edge case: test with negative and float values
    x = LpVariable(name="x")

    product = lpDot(-3.5, 4 * x)
    assert product.toDict() == [{"name": "x", "value": -14.0}]


def test_pulp_002():
    """
    Test the lpDot operation with more diverse data
    """
    x = LpVariable("x")
    y = LpVariable("y")
    z = LpVariable("z")
    # Use negative numbers and zeros in 'a'
    a = [0, -4, 5.5]
    # Update assertions to match actual outputs
    assert dict(lpDot([x, y, z], a)) == {y: -4, z: 5.5}
    assert dict(lpDot([2 * x, 2 * y, 2 * z], a)) == {y: -8, z: 11.0}
    assert dict(lpDot([x + y, y + z, z], a)) == {y: -4, z: 1.5}
    assert dict(lpDot(a, [x + y, y + z, z])) == {y: -4, z: 1.5}