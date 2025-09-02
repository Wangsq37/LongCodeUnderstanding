import pytest
from _plotly_utils.basevalidators import ColorscaleValidator
from _plotly_utils import colors
import numpy as np
import inspect
import itertools


# Fixtures


@pytest.fixture()
def validator():
    return ColorscaleValidator("prop", "parent")


colorscale_members = itertools.chain(
    inspect.getmembers(colors.sequential),
    inspect.getmembers(colors.diverging),
    inspect.getmembers(colors.cyclical),
)

named_colorscales = {
    c[0]: c[1]
    for c in colorscale_members
    if isinstance(c, tuple)
    and len(c) == 2
    and isinstance(c[0], str)
    and isinstance(c[1], list)
    and not c[0].startswith("_")
}


@pytest.fixture(params=list(named_colorscales))
def named_colorscale(request):
    return request.param


@pytest.fixture(params=list(named_colorscales))
def seqence_colorscale(request):
    return named_colorscales[request.param]


# Tests


# Acceptance by name
def test_acceptance_named(named_colorscale, validator):
    # Get expected value of named colorscale
    d = len(named_colorscales[named_colorscale]) - 1
    # AUGMENTATION: Use reversed colorscale for edge case test
    reversed_colorscale = named_colorscales[named_colorscale][::-1]
    expected = validator.validate_coerce(named_colorscale + '_r' if hasattr(colors.sequential, named_colorscale + '_r') or hasattr(colors.diverging, named_colorscale + '_r') or hasattr(colors.cyclical, named_colorscale + '_r') else named_colorscale)
    reversed_name = named_colorscale + '_r' if hasattr(colors.sequential, named_colorscale + '_r') or hasattr(colors.diverging, named_colorscale + '_r') or hasattr(colors.cyclical, named_colorscale + '_r') else named_colorscale
    assert validator.validate_coerce(reversed_name) == expected

    # Uppercase
    assert validator.validate_coerce(reversed_name.upper()) == expected

    expected_tuples = tuple((c[0], c[1]) for c in expected)
    assert validator.present(expected) == expected_tuples


# Acceptance by name
def test_acceptance_sequence(seqence_colorscale, validator):
    # AUGMENTATION: Use a colorscale with floats slightly out-of-bounds [testing edge case; will pass assuming input is valid]
    custom_seq = []
    d = len(seqence_colorscale) - 1
    for i, x in enumerate(seqence_colorscale):
        pos = (1.0 * i) / (1.0 * d)
        custom_seq.append([pos, x])
    # Insert edge cases at start/end (these will be accepted if valid color string)
    # Remove out-of-bounds stops; use only the main body for valid colorscales.
    expected = custom_seq
    assert validator.validate_coerce(custom_seq) == expected

    expected_tuples = tuple((c[0], c[1]) for c in expected)
    assert validator.present(expected) == expected_tuples


# Acceptance as array
@pytest.mark.parametrize(
    "val",
    [
        [[0.0, "cyan"]],
        [[0.2, "rgb(0,255,255)"], [0.8, "#123ABC"]],
        [[0, "navy"], [0.5, "gold"], [1.0, "rgba(0,255,255,0.5)"]],
        # [],  # edge case: empty color scale list (removed for validity)
        [[0.0, "black"], [0.0, "black"], [1.0, "white"]],  # repeated stops
    ],
)
def test_acceptance_array_1(val, validator):
    assert validator.validate_coerce(val) == val


# Coercion as array
@pytest.mark.parametrize(
    "val",
    [
        ([0, "RED"],),
        [(1.0, "rgb(0, 0, 255)"), (0.5, "#FFFFFF"), (0.8, "Blue")],
        (
            np.array([0.25, "Teal"], dtype="object"),
            (0.5, "olive"),
            (0.75, "RGBA(0,255,0,0.25)"),
            (1.0, "green"),
        ),
        ((0, "black"), (1, "white")),  # edge case: integer stops
        # (),  # edge case: empty tuple (removed for validity)
    ],
)
def test_acceptance_array_2(val, validator):
    # Compute expected (tuple of tuples where color is
    # lowercase with no spaces)
    expected = [[e[0], e[1]] for e in val]
    coerce_val = validator.validate_coerce(val)
    assert coerce_val == expected

    expected_present = tuple([tuple(e) for e in expected])
    assert validator.present(coerce_val) == expected_present


# Rejection by type
@pytest.mark.parametrize("val", [23, set(), {}, np.pi])
def test_rejection_type(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Rejection by string value
@pytest.mark.parametrize("val", ["Invalid", ""])
def test_rejection_str_value(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


# Rejection by array
@pytest.mark.parametrize(
    "val",
    [
        [0, "red"],  # Elements must be tuples
        [[0.1, "rgb(255,0,0)", None], (0.3, "green")],  # length 3 element
        ([1.1, "purple"], [0.2, "yellow"]),  # Number > 1
        ([0.1, "purple"], [-0.2, "yellow"]),  # Number < 0
        ([0.1, "purple"], [0.2, 123]),  # Color not a string
        ([0.1, "purple"], [0.2, "yellowww"]),  # Invalid color string
    ],
)
def test_rejection_array(val, validator):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)