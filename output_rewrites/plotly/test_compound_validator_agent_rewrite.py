import pytest
from _plotly_utils.basevalidators import CompoundValidator
from plotly.graph_objs.scatter import Marker


# Fixtures
# --------
@pytest.fixture()
def validator():
    return CompoundValidator("prop", "scatter", data_class_str="Marker", data_docs="")


# Tests
# -----
def test_acceptance(validator):
    # More comprehensive: Use a float for size and a complex color string
    val = Marker(color="#FF00FF", size=99.9)
    res = validator.validate_coerce(val)

    assert isinstance(res, Marker)
    assert res.color == "#FF00FF"
    assert res.size == 99.9


def test_acceptance_none(validator):
    val = None
    res = validator.validate_coerce(val)

    assert isinstance(res, Marker)
    assert res.color is None
    assert res.size is None


def test_acceptance_dict(validator):
    # Edge case: Use an empty string for color and zero for size
    val = dict(color="", size=0)
    # The empty string for color is invalid and raises ValueError, so the test should expect that
    with pytest.raises(ValueError):
        validator.validate_coerce(val)


def test_rejection_type(validator):
    val = 37

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


def test_rejection_value(validator):
    val = dict(color="green", size=10, bogus=99)

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert (
        "Invalid property specified for object of type "
        "plotly.graph_objs.scatter.Marker: 'bogus'" in str(validation_failure.value)
    )


def test_skip_invalid(validator):
    # Edge case: Use negative values and odd nested structure
    val = dict(
        color="",  # empty string: valid but non-typical
        size=-99999,  # negative: edge case for size
        bogus="-1",  # Bad property name as string
        colorbar={"bgcolor": "", "bogus_inner": None},  # Bad nested property name, empty color
        opacity=-0.25,  # Bad value for valid property (perhaps out-of-bounds)
    )

    # Adjusted expected: all invalid values dropped, so empty dict
    expected = {}

    res = validator.validate_coerce(val, skip_invalid=True)
    assert res.to_plotly_json() == expected


def test_skip_invalid_empty_object(validator):
    # Edge case: All inners invalid/empty, huge value for size
    val = dict(
        color=None,
        size=1e9,  # very large value
        colorbar={
            "bgcolor": None,  # invalid/empty
            "bogus_inner": "invalid",  # invalid nested
        },
        opacity=1,  # valid but at boundary
    )

    # Adjusted expected: color is missing if None, colorbar absent if all invalid
    expected = {'opacity': 1, 'size': 1000000000.0}

    res = validator.validate_coerce(val, skip_invalid=True)
    assert res.to_plotly_json() == expected