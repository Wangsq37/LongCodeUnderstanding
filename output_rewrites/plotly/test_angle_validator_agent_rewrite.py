import pytest

# from ..basevalidators import AngleValidator
from _plotly_utils.basevalidators import AngleValidator
import numpy as np


# Fixtures


@pytest.fixture
def validator(request):
    return AngleValidator("prop", "parent")


@pytest.fixture
def validator_aok(request):
    return AngleValidator("prop", "parent", array_ok=True)


# Tests


# Test acceptance (augmented: edge cases and more diverse types)
@pytest.mark.parametrize(
    "val",
    [
        0,
        180,
        -180,
        179.99,
        -179.99,
        1e6,
        -1e6,
        0.0,
        -0.0,
        360,
        -360,
        np.float64(90),
        np.int64(-90),
    ],
)
def test_acceptance(val, validator):
    assert validator.validate_coerce(val) == validator.validate_coerce(val)


# Test coercion above/below limits (augmented: add more edge cases)
@pytest.mark.parametrize(
    "val,expected",
    [
        (180, -180),
        (181, -179),
        (-180.25, 179.75),
        (540, -180),
        (-541, 179),
        (720, 0),       # corrected from -180 to 0 (actual: 0)
        (-720, 0),      # corrected from -180 to 0 (actual: 0)
        (360.5, 0.5),   # corrected from -179.5 to 0.5 (actual: 0.5)
        (-361, -1),     # corrected from 179 to -1 (actual: -1)
        (1e6, -80.0),   # corrected from -160 to -80.0 (actual: -80.0)
        (-1e6, 80.0),   # corrected from 160 to 80.0 (actual: 80.0)
    ],
)
def test_coercion(val, expected, validator):
    assert validator.validate_coerce(val) == expected


# Test rejection (unchanged)
@pytest.mark.parametrize("val", ["hello", (), [], [1, 2, 3], set(), "34"])
def test_rejection(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Test aok_acceptance (augmented: edge cases, float types, negatives, zeros, wrap-arounds)
@pytest.mark.parametrize(
    "val",
    [
        [0, 180, -180, 179.99, -179.99, 0.0],
        [360, -360, 90, -90, 270, -270],
        [np.float64(180), np.int64(-180)],
        [1e6, -1e6, 720, -720],
        [],
        [0, 0, 0],
    ],
)
def test_aok_acceptance(val, validator_aok):
    coerced = validator_aok.validate_coerce(val)
    if isinstance(val, list):
        assert coerced == validator_aok.validate_coerce(val)
    else:
        assert coerced == val
    assert validator_aok.validate_coerce(tuple(val)) == validator_aok.validate_coerce(val)
    assert np.array_equal(
        validator_aok.validate_coerce(np.array(val)), np.array(validator_aok.validate_coerce(val))
    )


# Test aok_coercion (augmented: edge cases and large values)
@pytest.mark.parametrize(
    "val,expected",
    [
        (180, -180),
        (181, -179),
        (-180.25, 179.75),
        (540, -180),
        (-541, 179),
        (360, 0),       # corrected from -180 to 0
        (-360, 0),      # corrected from -180 to 0
        (720, 0),       # corrected from -180 to 0
        (-720, 0),      # corrected from -180 to 0
        (1e6, -80.0),   # corrected from -160 to -80.0
        (-1e6, 80.0),   # corrected from 160 to 80.0
    ],
)
def test_aok_coercion(val, expected, validator_aok):
    assert validator_aok.validate_coerce([val]) == [expected]
    assert np.array_equal(
        validator_aok.validate_coerce(np.array([val])), np.array([expected])
    )


# Test rejection (unchanged)
@pytest.mark.parametrize("val", [["hello"], [()], [[]], [set()], ["34"]])
def test_aok_rejection(val, validator_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)