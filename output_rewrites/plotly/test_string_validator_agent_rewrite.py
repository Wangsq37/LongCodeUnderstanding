import pytest

from _plotly_utils.basevalidators import StringValidator
import numpy as np
from ...test_optional.test_utils.test_utils import np_nan


# Fixtures


@pytest.fixture()
def validator():
    return StringValidator("prop", "parent")


@pytest.fixture()
def validator_values():
    return StringValidator("prop", "parent", values=["foo", "BAR", ""])


@pytest.fixture()
def validator_no_blanks():
    return StringValidator("prop", "parent", no_blank=True)


@pytest.fixture()
def validator_strict():
    return StringValidator("prop", "parent", strict=True)


@pytest.fixture
def validator_aok():
    return StringValidator("prop", "parent", array_ok=True, strict=False)


@pytest.fixture
def validator_aok_strict():
    return StringValidator("prop", "parent", array_ok=True, strict=True)


@pytest.fixture
def validator_aok_values():
    return StringValidator(
        "prop", "parent", values=["foo", "BAR", "", "baz"], array_ok=True
    )


@pytest.fixture()
def validator_no_blanks_aok():
    return StringValidator("prop", "parent", no_blank=True, array_ok=True)


# Array not ok


# Not strict
# Acceptance
@pytest.mark.parametrize(
    "val", ["abc123", -42, 0, "LARGE_STRING_"*100, "测试", " ", None, 1.5e10, "\u00a0", "\n", "\t"]
)
def test_acceptance(val, validator):
    expected = val if val is None else (str(val) if not isinstance(val, str) else val)
    assert validator.validate_coerce(val) == expected


# Rejection by value
@pytest.mark.parametrize("val", [(), [], [1, 2, 3], set()])
def test_rejection(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Valid values


@pytest.mark.parametrize("val", ["BAR", "", "foo"])
def test_acceptance_values(val, validator_values):
    assert validator_values.validate_coerce(val) == val


@pytest.mark.parametrize("val", ["FOO", "bar", "other", "1234"])
def test_rejection_values(val, validator_values):
    with pytest.raises(ValueError) as validation_failure:
        validator_values.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)
    assert "['foo', 'BAR', '']" in str(validation_failure.value)


# No blanks
@pytest.mark.parametrize("val", ["foo", "   ", "\u00a0", "0", "non-empty"])
def test_acceptance_no_blanks(val, validator_no_blanks):
    assert validator_no_blanks.validate_coerce(val) == val


@pytest.mark.parametrize("val", [""])
def test_rejection_no_blanks(val, validator_no_blanks):
    with pytest.raises(ValueError) as validation_failure:
        validator_no_blanks.validate_coerce(val)

    assert "A non-empty string" in str(validation_failure.value)


# Strict


# Acceptance
@pytest.mark.parametrize("val", ["foo", "   ", "0", "non-empty", "\u00a0"])
def test_acceptance_strict(val, validator_strict):
    assert validator_strict.validate_coerce(val) == val


# Rejection by value
@pytest.mark.parametrize("val", [(), [], [1, 2, 3], set(), np_nan(), np.pi, 23])
def test_rejection_strict(val, validator_strict):
    with pytest.raises(ValueError) as validation_failure:
        validator_strict.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Array ok


# Acceptance
@pytest.mark.parametrize("val", ["foo", "BAR", "", "baz", "\u03bc", "LARGE_STRING_"*50])
def test_acceptance_aok_scalars(val, validator_aok):
    assert validator_aok.validate_coerce(val) == val


@pytest.mark.parametrize(
    "val",
    [
        ["foo", "BAR", "baz", "extra", "123"],
        np.array(["test", "value", "\u03bc", "0", "abc"], dtype="object"),
        ["non-empty", "LARGE_STRING_"*10],
        ["foo", None, "bar", "\u00a0", ""],
        ["BAR", "", "baz", "foo"],
    ],
)
def test_acceptance_aok_list(val, validator_aok):
    coerce_val = validator_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert isinstance(coerce_val, np.ndarray)
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, list):
        assert validator_aok.present(val) == tuple(val)
    else:
        assert coerce_val == val


# Rejection by type
@pytest.mark.parametrize("val", [["foo", ()], ["foo", 3, 4], [3, 2, 1]])
def test_rejection_aok(val, validator_aok_strict):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_strict.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# Rejection by value
@pytest.mark.parametrize(
    "val", [["foo", "bar"], ["3", "4"], ["BAR", "BAR", "hello!"], ["foo", None]]
)
def test_rejection_aok_values(val, validator_aok_values):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_values.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# No blanks
@pytest.mark.parametrize(
    "val",
    [
        "not_blank",
        ["non-empty", "space"],
        np.array(["symbol", "name", "\u20ac"], dtype="object"),
        ["foo", "bar", "baz", "qux"],
    ],
)
def test_acceptance_no_blanks_aok(val, validator_no_blanks_aok):
    coerce_val = validator_no_blanks_aok.validate_coerce(val)
    if isinstance(val, np.ndarray):
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, list):
        assert validator_no_blanks_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val


@pytest.mark.parametrize(
    "val",
    [
        "",
        ["foo", "bar", ""],
        np.array(["foo", "bar", ""], dtype="object"),
        [""],
        np.array([""], dtype="object"),
    ],
)
def test_rejection_no_blanks_aok(val, validator_no_blanks_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_no_blanks_aok.validate_coerce(val)

    assert "A non-empty string" in str(validation_failure.value)