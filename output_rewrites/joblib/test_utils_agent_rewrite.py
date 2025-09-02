import pytest

from joblib._utils import eval_expr


@pytest.mark.parametrize(
    "expr",
    ["exec('import os')", "print(1)", "import os", "1+1; import os", "1^1"],
)
def test_eval_expr_invalid(expr):
    with pytest.raises(ValueError, match="is not a valid or supported arithmetic"):
        eval_expr(expr)


@pytest.mark.parametrize(
    "expr, result",
    [
        ("0*123456789", 0),                            # multiplying by zero
        ("(-5) + 3", -2),                              # negative addition
        ("1000000 // 3", 333333),                      # large integer division
        ("2.5 * 4 - 7.5", 2.5),                        # float arithmetic
        ("(7 % 3) ** 4", 1),                           # modulus and power
        ("(0 - 2) * (3 + 4)", -14),                    # mixture of negation and grouping
        ("(6 + 2.4) / 2", 4.2),                        # float result division
        ("(10 // 3) * (-2)", -6),                      # negative multiply with floor division
        ("(1.5 + 2.5) * 2", 8.0),                      # decimal addition then multiply
        ("0 ** 10", 0),                                # zero power positive
    ],
)
def test_eval_expr_valid(expr, result):
    assert eval_expr(expr) == result