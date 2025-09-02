import pytest
from _plotly_utils.basevalidators import ColorValidator
import numpy as np


# Fixtures


@pytest.fixture()
def validator():
    return ColorValidator("prop", "parent")


@pytest.fixture()
def validator_colorscale():
    return ColorValidator("prop", "parent", colorscale_path="parent.colorscale")


@pytest.fixture()
def validator_aok():
    return ColorValidator("prop", "parent", array_ok=True)


@pytest.fixture()
def validator_aok_colorscale():
    return ColorValidator(
        "prop", "parent", array_ok=True, colorscale_path="parent.colorscale"
    )


# Array not ok, numbers not ok
@pytest.mark.parametrize(
    "val",
    [
        "",  # Edge case: empty string
        "YELLOW",
        "rgba(0, 0, 0, 0)",
        "var(--primary-color)",
        "hsl(360, 100%, 100%)",
        "hsla(360, 50%, 50%, 0.5)",
        "hsv(120, 56%, 67%)",
        "hsva(240, 30%, 45%, 0.3)",
    ],
)
def test_acceptance(val, validator):
    if val == "":
        with pytest.raises(ValueError):
            validator.validate_coerce(val)
    else:
        assert validator.validate_coerce(val) == val


# Rejection by type
@pytest.mark.parametrize("val", [set(), 23, 0.5, {}, ["red"], [12]])
def test_rejection_1(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Rejection by value
@pytest.mark.parametrize("val", ["redd", "rgbbb(255, 0, 0)", "hsl(0, 1%0000%, 50%)"])
def test_rejection_2(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Array not ok, numbers ok


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        "CYAN",
        0,            # Edge case: zero
        -17,          # Edge case: negative integer
        1000000,      # Large integer
        "rgba(100, 200, 10, 0.5)",
        "var(--secondary-color)",
        "hsl(180, 60%, 35%)",
        "hsla(90, 70%, 80%, 0.8)",
        "hsv(180, 50%, 100%)",
        "hsva(60, 40%, 90%, 1.0)",
    ],
)
def test_acceptance_colorscale(val, validator_colorscale):
    assert validator_colorscale.validate_coerce(val) == val


# Rejection by type
@pytest.mark.parametrize("val", [set(), {}, ["red"], [12]])
def test_rejection_colorscale_1(val, validator_colorscale):
    with pytest.raises(ValueError) as validation_failure:
        validator_colorscale.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Rejection by value
@pytest.mark.parametrize("val", ["redd", "rgbbb(255, 0, 0)", "hsl(0, 1%0000%, 50%)"])
def test_rejection_colorscale_2(val, validator_colorscale):
    with pytest.raises(ValueError) as validation_failure:
        validator_colorscale.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Array ok, numbers not ok


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        "pink",
        ["blue", "rgba(0,0,0,0)"],
        np.array(["blue", "rgba(0,0,0,0)", "", "hsl(360,100%,100%)"]),
        ["hsl(180, 60%, 35%)", "rgba(0,0,0,0)", "hsv(180,50%,100%)"],
        np.array(["hsl(180, 60%, 35%)", "rgba(0,0,0,0)", "hsv(180, 50%, 100%)"]),
        ["hsva(60,40%,90%,1.0)"],
        [],  # Edge case: empty list
    ],
)
def test_acceptance_aok(val, validator_aok):
    if isinstance(val, np.ndarray):
        if "" in val:
            with pytest.raises(ValueError):
                validator_aok.validate_coerce(val)
        else:
            coerce_val = validator_aok.validate_coerce(val)
            assert np.array_equal(coerce_val, val)
    elif isinstance(val, list):
        if "" in val:
            with pytest.raises(ValueError):
                validator_aok.validate_coerce(val)
        else:
            coerce_val = validator_aok.validate_coerce(val)
            assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert validator_aok.validate_coerce(val) == val


@pytest.mark.parametrize(
    "val",
    [
        "purple",
        [["green", "rgba(0,0,0,0)"]],
        [["hsl(0, 0%, 50%)", "hsla(0, 0%, 50%, 0.5)"], ["hsv(10, 100%, 10%)", "blue"]],
        np.array([
            ["rgba(0,0,0,0)", "blue"],
            ["hsl(0, 0%, 50%)", "hsla(0, 0%, 50%, 0.5)"],
        ]),
        [[]],  # Edge case: 2D list containing empty list
    ],
)
def test_acceptance_aok_2D(val, validator_aok):
    coerce_val = validator_aok.validate_coerce(val)

    if isinstance(val, np.ndarray):
        assert np.array_equal(coerce_val, val)
    elif isinstance(val, list):
        assert validator_aok.present(coerce_val) == tuple(val)
    else:
        assert coerce_val == val


# Rejection
@pytest.mark.parametrize(
    "val",
    [
        [23],
        [0, 1, 2],
        ["redd", "rgb(255, 0, 0)"],
        ["hsl(0, 100%, 50_00%)", "hsla(0, 100%, 50%, 100%)", "hsv(0, 100%, 100%)"],
        ["hsva(0, 1%00%, 100%, 50%)"],
    ],
)
def test_rejection_aok(val, validator_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


@pytest.mark.parametrize(
    "val",
    [
        [["redd", "rgb(255, 0, 0)"]],
        [
            ["hsl(0, 100%, 50_00%)", "hsla(0, 100%, 50%, 100%)"],
            ["hsv(0, 100%, 100%)", "purple"],
        ],
        [
            np.array(["hsl(0, 100%, 50_00%)", "hsla(0, 100%, 50%, 100%)"]),
            np.array(["hsv(0, 100%, 100%)", "purple"]),
        ],
        [["blue"], [2]],
    ],
)
def test_rejection_aok_2D(val, validator_aok):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# Array ok, numbers ok


# Acceptance
@pytest.mark.parametrize(
    "val",
    [
        "black",
        -100,                 # Edge case: negative integer
        [0, -1, 1e6, 3.1415], # Array with edge-case numbers
        ["green", 1.5, "rgba(255, 255, 0, 0.25)", ""], # Strings, floats, empty string
        ["hsl(359, 100%, 0%)", "hsla(0, 0%, 0%, 0)", "hsv(360, 100%, 100%)"],
        ["hsva(360, 100%, 100%, 0.0)", ""],
        np.array([1, 2, 3]),   # Array of numbers (edge case: numpy ints)
    ],
)
def test_acceptance_aok_colorscale(val, validator_aok_colorscale):
    if isinstance(val, (list, np.ndarray)):
        if "" in val:
            with pytest.raises(ValueError):
                validator_aok_colorscale.validate_coerce(val)
        else:
            coerce_val = validator_aok_colorscale.validate_coerce(val)
            assert np.array_equal(list(coerce_val), val)
    else:
        assert validator_aok_colorscale.validate_coerce(val) == val


# Rejection
@pytest.mark.parametrize(
    "val",
    [
        ["redd", 0.5, "rgb(255, 0, 0)"],
        ["hsl(0, 100%, 50_00%)", "hsla(0, 100%, 50%, 100%)", "hsv(0, 100%, 100%)"],
        ["hsva(0, 1%00%, 100%, 50%)"],
    ],
)
def test_rejection_aok_colorscale(val, validator_aok_colorscale):
    with pytest.raises(ValueError) as validation_failure:
        validator_aok_colorscale.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# Description


# Test dynamic description logic
def test_description(validator):
    desc = validator.description()
    assert "A number that will be interpreted as a color" not in desc
    assert "A list or array of any of the above" not in desc


def test_description_aok(validator_aok):
    desc = validator_aok.description()
    assert "A number that will be interpreted as a color" not in desc
    assert "A list or array of any of the above" in desc


def test_description_aok_colorscale(validator_aok_colorscale):
    desc = validator_aok_colorscale.description()
    assert "A number that will be interpreted as a color" in desc
    assert "A list or array of any of the above" in desc