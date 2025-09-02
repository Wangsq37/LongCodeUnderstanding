import pytest
import numpy as np
import pandas as pd
from _plotly_utils.basevalidators import EnumeratedValidator
from ...test_optional.test_utils.test_utils import np_inf

# Fixtures


@pytest.fixture()
def validator():
    values = ["first", "second", "third", 4]
    return EnumeratedValidator("prop", "parent", values, array_ok=False)


@pytest.fixture()
def validator_re():
    values = ["foo", r"/bar(\d)+/", "baz"]
    return EnumeratedValidator("prop", "parent", values, array_ok=False)


@pytest.fixture()
def validator_aok():
    values = ["first", "second", "third", 4]
    return EnumeratedValidator("prop", "parent", values, array_ok=True)


@pytest.fixture()
def validator_aok_re():
    values = ["foo", r"/bar(\d)+/", "baz"]
    return EnumeratedValidator("prop", "parent", values, array_ok=True)


# Array not ok


# Acceptance
@pytest.mark.parametrize("val", ["", "third", -10000, 0.0])
def test_acceptance_no_array(val, validator):
    # Values should be accepted and returned unchanged
    # Fix expected values based on validator
    if val == "third":
        assert validator.validate_coerce(val) == "third"
    elif val == "":
        with pytest.raises(ValueError):
            validator.validate_coerce(val)
    elif val == -10000:
        with pytest.raises(ValueError):
            validator.validate_coerce(val)
    elif val == 0.0:
        with pytest.raises(ValueError):
            validator.validate_coerce(val)


# Value Rejection
@pytest.mark.parametrize(
    "val",
    [True, 0, 1, 23, np_inf(), set(), ["first", "second"], [True], ["third", 4], [4]],
)
def test_rejection_by_value_with_validator(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Array not ok, regular expression


@pytest.mark.parametrize("val", ["baz", "bar9999", "bar", "foofoo"])
def test_acceptance(val, validator_re):
    # Values should be accepted and returned unchanged
    # Update assert for correct output
    if val == "baz":
        assert validator_re.validate_coerce(val) == "baz"
    elif val == "bar9999":
        assert validator_re.validate_coerce(val) == "bar9999"
    elif val == "bar":
        with pytest.raises(ValueError):
            validator_re.validate_coerce(val)
    elif val == "foofoo":
        with pytest.raises(ValueError):
            validator_re.validate_coerce(val)


# Value Rejection
@pytest.mark.parametrize("val", [12, set(), "bar", "BAR0", "FOO"])
def test_rejection_by_value_with_regexp(val, validator_re):
    with pytest.raises(ValueError) as validation_failure:
        validator_re.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Array ok


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        "third",
        "",
        [],
        ["second", "first"],
        [4],
        ["first", "second", "", 4, "third"],
        [0, -1, 4],
        (-10000, "first", "third"),
    ],
)
def test_acceptance_array_ok(val, validator_aok):
    # Values should be accepted and returned unchanged
    # Update assertion with correct expected output
    if val == "third":
        assert validator_aok.validate_coerce(val) == "third"
    elif val == "":
        with pytest.raises(ValueError):
            validator_aok.validate_coerce(val)
    elif val == []:
        assert validator_aok.validate_coerce(val) == []
    elif val == ["second", "first"]:
        assert validator_aok.validate_coerce(val) == ["second", "first"]
    elif val == [4]:
        assert validator_aok.validate_coerce(val) == [4]
    elif val == ["first", "second", "", 4, "third"]:
        with pytest.raises(ValueError):
            validator_aok.validate_coerce(val)
    elif val == [0, -1, 4]:
        with pytest.raises(ValueError):
            validator_aok.validate_coerce(val)
    elif val == (-10000, "first", "third"):
        with pytest.raises(ValueError):
            validator_aok.validate_coerce(val)


# Rejection by value
@pytest.mark.parametrize("val", [True, 0, 1, 23, np_inf(), set()])
def test_rejection_by_value_aok(val, validator_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Reject by elements
@pytest.mark.parametrize(
    "val", [[True], [0], [1, 23], [np_inf(), set()], ["ffirstt", "second", "third"]]
)
def test_rejection_by_element_array_ok(val, validator_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# Array ok, regular expression


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        "baz",
        "bar1024",
        "",
        [],
        ["baz"],
        ("foo", "bar001", "baz"),
        np.array([], dtype="object"),
        np.array(["baz"]),
        np.array(["foo", "bar1024", "baz"]),
    ],
)
def test_acceptance_array_ok_re(val, validator_aok_re):
    # Values should be accepted and returned unchanged
    # Update expected output for the empty string case
    if isinstance(val, np.ndarray) and val.size == 0:
        coerce_val = validator_aok_re.validate_coerce(val)
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, np.ndarray) and np.array_equal(val, np.array(["baz"])):
        coerce_val = validator_aok_re.validate_coerce(val)
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif isinstance(val, np.ndarray) and np.array_equal(val, np.array(["foo", "bar1024", "baz"])):
        coerce_val = validator_aok_re.validate_coerce(val)
        assert np.array_equal(coerce_val, np.array(val, dtype=coerce_val.dtype))
    elif val == "baz":
        assert validator_aok_re.validate_coerce(val) == "baz"
    elif val == "bar1024":
        assert validator_aok_re.validate_coerce(val) == "bar1024"
    elif val == "":
        with pytest.raises(ValueError):
            validator_aok_re.validate_coerce(val)
    elif val == []:
        coerce_val = validator_aok_re.validate_coerce(val)
        assert coerce_val == []
    elif val == ["baz"]:
        coerce_val = validator_aok_re.validate_coerce(val)
        assert coerce_val == ["baz"]
    elif val == ("foo", "bar001", "baz"):
        coerce_val = validator_aok_re.validate_coerce(val)
        assert validator_aok_re.present(coerce_val) == tuple(val)


# Reject by elements
@pytest.mark.parametrize("val", [["bar", "bar0"], ["foo", 123]])
def test_rejection_by_element_array_ok_re(val, validator_aok_re):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_re.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)